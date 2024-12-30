
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QComboBox, QWidget, QPushButton, QMessageBox
from PyQt6.QtGui import QStandardItemModel, QStandardItem

from PyQt6.QtCore import Qt, pyqtSignal
import pyqtgraph as pg
import numpy as np

from scipy.interpolate import CubicSpline

import betz

def fit_quadratic(x, y):
    A = np.array([
        [x[0]**2, x[0], 1],
        [x[1]**2, x[1], 1],
        [x[2]**2, x[2], 1]
    ])
    b = np.array(y)
    coeffs = np.linalg.solve(A, b)
    return coeffs

class DraggableScatterPlotItem(pg.ScatterPlotItem):
    def __init__(self, control_points, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_points = control_points
        self.dragged_point_index = None
        self.is_spline = False
        self.finished_dragging = False

    def mouseDragEvent(self, ev):
        
        if ev.button() != pg.QtCore.Qt.MouseButton.LeftButton:
            ev.ignore()
            return

        if not self.isVisible():
            ev.ignore()
            return

        if ev.isStart():
            pos = ev.pos()
            distances = [pg.Point(pos - pg.Point(p)).length() for p in self.control_points]
            threshold = 0.05 * max(self.viewRect().width(), 
                                   self.viewRect().height())
            if min(distances) > threshold:
                if self.is_spline:
                    self.control_points = np.vstack([self.control_points, [pos.x(), pos.y()]])
                    self.dragged_point_index = self.control_points.shape[0] - 1
                else:
                    ev.ignore()
                    return
            else:
                self.dragged_point_index = np.argmin(distances)
            ev.accept()
        elif ev.isFinish():
            self.dragged_point_index = None
            self.finished_dragging = True
            self.sigPlotChanged.emit(self)
        else:
            if self.dragged_point_index is not None:
                # Update the position of the dragged point
                self.control_points[self.dragged_point_index] = [ev.pos().x(), ev.pos().y()]
                self.setData(pos=self.control_points)
                self.sigPlotChanged.emit(self)
    
    # double click removes point
    def mouseClickEvent(self, ev):
        if ev.button() != pg.QtCore.Qt.MouseButton.LeftButton:
            ev.ignore()
            return
        
        if not ev.double():
            ev.ignore()
            return
        
        if not self.is_spline:
            ev.ignore()
            return
        
        if self.control_points.shape[0] < 3:
            ev.ignore()
            return
        
        pos = ev.pos()
        distances = [pg.Point(pos - pg.Point(p)).length() for p in self.control_points]
        threshold = 0.05 * max(self.viewRect().width(), 
                                self.viewRect().height())
        if min(distances) < threshold:
            self.control_points = np.delete(self.control_points, np.argmin(distances), axis=0)
            self.setData(pos=self.control_points)
            self.sigPlotChanged.emit(self)

class DistributionPlotWidget(QWidget):
    new_dist = pyqtSignal()

    def __init__(self, parent = None, title = None, xlabel = None, ylabel = None):
        super().__init__()
        
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        self.dist_type = QComboBox()
        self.dist_model = QStandardItemModel(self.dist_type)
        #self.dist_type.addItems(["Linear", "Quadratic", "Spline", "Betz"])
        self.dist_model.appendRow(QStandardItem("Linear"))
        self.dist_model.appendRow(QStandardItem("Quadratic"))
        self.dist_model.appendRow(QStandardItem("Spline"))
        custom_item = QStandardItem("Custom")
        custom_item.setFlags( custom_item.flags() & ~Qt.ItemFlag.ItemIsSelectable )
        self.dist_model.appendRow(custom_item)
        self.dist_type.setModel(self.dist_model)
        self.dist_type.currentIndexChanged.connect(self.update_distribution)
        
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setXRange(0, 1)
        self.plot_widget.setYRange(0, 1)
        self.plot_widget.setLabel('left', 'Value')
        self.plot_widget.setLabel('bottom', 'Radius')

        if title is not None:
            self.plot_widget.setTitle(title)
        if xlabel is not None:
            self.plot_widget.setLabel('bottom', xlabel)
        if ylabel is not None:
            self.plot_widget.setLabel('left', ylabel)
        self.plot_widget.showGrid(x=True, y=True)

        layout.addWidget(self.plot_widget)
        layout.addWidget(self.dist_type)
        
        # initially linear
        control_points = np.array([[0, 0], [1, 1]]).astype(float)  # Two points for linear by default
        self.scatter = DraggableScatterPlotItem(
            control_points, size=10, brush=pg.mkBrush(255, 0, 0), pen=pg.mkPen(None)
        )
        self.scatter.sigPlotChanged.connect(self.update_curve)

        self.plot_widget.addItem(self.scatter)
        
        self.update_distribution(0)

        self.xb = np.linspace(0, 1, 100)
        self.yb = np.zeros_like(self.xb)
    
    def update_distribution(self, index):
        self.scatter.is_spline = False
        self.scatter.show()

        if index == 0:  # linear
            self.set_distribution( "linear",
                    np.array([[0.0, 0.1], [1.0, 0.1]])
                )
        elif index == 1:  # quadratic
            self.set_distribution( "quadratic",
                np.array([[0.0, 0.1], [0.5, 0.2], [1.0, 0.1]])
            )
        elif index == 2: # spline
            self.set_distribution( "spline",
                np.array([[0.0, 0.1], [0.5, 0.2], [1.0, 0.1]])
                )
        elif index == 3: # custom
            self.set_distribution( "custom",
                np.array([[0.0, 0.1], [0.5, 0.2], [1.0, 0.1]])
                )

    def get_distribution(self, x_dist):
        x = self.scatter.control_points[:, 0]
        y = self.scatter.control_points[:, 1]

        index = self.dist_type.currentIndex()

        if index == 0:
            b = (y[1] - y[0]) / (x[1] - x[0])
            a = y[0] - b * x[0]
            y_dist = a + b * x_dist
        elif index == 1:
            coefficients = fit_quadratic(x, y)
            y_dist = coefficients[0] * x_dist**2 + coefficients[1] * x_dist + coefficients[2]
        elif index == 2:
            idx = np.argsort(x)
            spline = CubicSpline(x[idx], y[idx])
            y_dist = spline(x_dist)
        elif index == 3:
            return self.yb
        else:
            y_dist = np.zeros_like(x_dist)

        return y_dist
    
    def update_curve(self):

        index = self.dist_type.currentIndex()
        if index == 3:
            x_plot = self.xb
            y_plot = self.yb
        else:
            x_plot = np.linspace(0, 1, 100)
            y_plot = self.get_distribution(x_plot)
        
        self.plot_widget.clear()
        self.plot_widget.addItem(self.scatter)
        self.plot_widget.plot(x_plot, y_plot, pen='b')

        if self.scatter.finished_dragging:
            self.scatter.finished_dragging = False
            
            # if control points are outside of the plot range move the plot range
            ymin, ymax = self.plot_widget.viewRange()[1]
            ycmin = np.min(self.scatter.control_points[:, 1])
            ycmax = np.max(self.scatter.control_points[:, 1])
            update = False
            if ycmin < ymin:
                ymin = ycmin - 0.1
                update = True
            if ycmax > ymax:
                ymax = ycmax + 0.1
                update = True
            if update:
                self.plot_widget.setYRange(ymin, ymax)

        self.new_dist.emit()
    
    def set_distribution(self, distype, *args):
        distype = distype.strip().lower()
        self.scatter.is_spline = False
        self.scatter.show()

        self.dist_type.currentIndexChanged.disconnect(self.update_distribution)
        
        if distype == "linear":
            self.dist_type.setCurrentIndex(0)
            self.scatter.control_points = args[0]
        elif distype == "quadratic":
            self.dist_type.setCurrentIndex(1)
            self.scatter.control_points = args[0]
        elif distype == "spline":
            self.dist_type.setCurrentIndex(2)
            self.scatter.control_points = args[0]
            self.scatter.is_spline = True
        elif distype == "custom":
            self.dist_type.setCurrentIndex(3)
            self.scatter.control_points = np.zeros((0, 2))
            self.xb = args[0]
            self.yb = args[1]
            self.scatter.hide()
        
        self.scatter.setData(pos=self.scatter.control_points)

        self.dist_type.currentIndexChanged.connect(self.update_distribution)

class DistributionsWidget(QWidget):
    new_dist = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        
        layout = QVBoxLayout(self)

        self.betz_button = QPushButton("Betz Optimised")
        self.betz_button.clicked.connect(self.betz_optimised)
        layout.addWidget(self.betz_button)

        self.avs = None
        
        self.chord_plot = DistributionPlotWidget(self, title="Chord", ylabel="Chord [m]")
        self.twist_plot = DistributionPlotWidget(self, title="Twist", ylabel="Twist [rad]")
        self.sweep_plot = DistributionPlotWidget(self, title="Sweep", ylabel="Sweep [rad]")

        layout.addWidget(self.chord_plot)
        layout.addWidget(self.twist_plot)
        layout.addWidget(self.sweep_plot)

        self.attach_dist_signals()

        
    def attach_dist_signals(self):
        self.chord_plot.new_dist.connect(self.on_new_dist)
        self.sweep_plot.new_dist.connect(self.on_new_dist)
        self.twist_plot.new_dist.connect(self.on_new_dist)
    
    def detach_dist_signals(self):
        self.chord_plot.new_dist.disconnect(self.on_new_dist)
        self.sweep_plot.new_dist.disconnect(self.on_new_dist)
        self.twist_plot.new_dist.disconnect(self.on_new_dist)

    def on_new_dist(self):
        self.new_dist.emit()

    def update_avs(self, avs):

        try:
            avs.prop['r0_rt']
        except KeyError:
            return

        # high resolution distributions
        avs.prop['c'] = self.chord_plot.get_distribution(avs.prop['r0_rt'])
        avs.prop['twist'] = self.twist_plot.get_distribution(avs.prop['r0_rt'])
        avs.prop['sweep'] = self.sweep_plot.get_distribution(avs.prop['r0_rt'])

        distypes = [
            "linear",
            "quadratic",
            "spline",
            "custom"
        ]

        avs.dist['CTL_c'] = self.chord_plot.scatter.control_points
        avs.dist['CTL_c_type'] = distypes[
            self.chord_plot.dist_type.currentIndex()]
        avs.dist['CTL_twist'] = self.twist_plot.scatter.control_points
        avs.dist['CTL_twist_type'] = distypes[
            self.twist_plot.dist_type.currentIndex()]
        avs.dist['CTL_sweep'] = self.sweep_plot.scatter.control_points
        avs.dist['CTL_sweep_type'] = distypes[
            self.sweep_plot.dist_type.currentIndex()]

        self.avs = avs
    
    def betz_optimised(self):
        if self.avs is None:
            return
        
        if betz.betz_design(self.avs):
            self.set_dists(self.avs)

            betz.betz_off_design(self.avs)

        else:
            QMessageBox.critical(
                self, "Error", "Betz Optimisation Failed", QMessageBox.StandardButton.Ok
            )


    def set_dists(self, avs):
        self.detach_dist_signals()

        
        chord_type = avs.dist['CTL_c_type']
        if chord_type == "custom":
            self.chord_plot.set_distribution(
                "custom", avs.prop['r0_rt'], avs.prop['c']
            )
        else:
            self.chord_plot.set_distribution(
                avs.dist['CTL_c_type'], avs.dist['CTL_c']
            )
        
        twist_type = avs.dist['CTL_twist_type']
        if twist_type == "custom":
            self.twist_plot.set_distribution(
                "custom", avs.prop['r0_rt'], avs.prop['twist']
            )
        else:
            self.twist_plot.set_distribution(
                avs.dist['CTL_twist_type'], avs.dist['CTL_twist']
            )
        
        sweep_type = avs.dist['CTL_sweep_type']
        if sweep_type == "custom":
            self.sweep_plot.set_distribution(
                "custom", avs.prop['r0_rt'], avs.prop['sweep']
            )
        else:
            self.sweep_plot.set_distribution(
                avs.dist['CTL_sweep_type'], avs.dist['CTL_sweep']
            )

        self.attach_dist_signals()
        self.new_dist.emit()



