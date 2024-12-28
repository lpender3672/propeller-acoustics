
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QComboBox, QWidget
from PyQt6.QtCore import Qt, pyqtSignal
import pyqtgraph as pg
import numpy as np

from scipy.interpolate import CubicSpline

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

    def mouseDragEvent(self, ev):
        if ev.button() != pg.QtCore.Qt.MouseButton.LeftButton:
            ev.ignore()
            return

        if ev.isStart():
            pos = ev.pos()
            distances = [pg.Point(pos - pg.Point(p)).length() for p in self.control_points]
            if min(distances) > 0.05:
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
        else:
            if self.dragged_point_index is not None:
                # Update the position of the dragged point
                self.control_points[self.dragged_point_index] = [ev.pos().x(), ev.pos().y()]
                self.setData(pos=self.control_points)
                self.sigPlotChanged.emit(self)

class DistributionPlotWidget(QWidget):
    new_dist = pyqtSignal()

    def __init__(self):
        super().__init__()
        
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        self.dist_type = QComboBox()
        self.dist_type.addItems(["Linear", "Quadratic", "Spline"])
        self.dist_type.currentIndexChanged.connect(self.update_distribution)
        layout.addWidget(self.dist_type)
        
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setXRange(0, 1)
        self.plot_widget.setYRange(0, 1)
        self.plot_widget.setLabel('left', 'Value')
        self.plot_widget.setLabel('bottom', 'Radius')
        self.plot_widget.showGrid(x=True, y=True)
        layout.addWidget(self.plot_widget)
        
        # initially linear
        control_points = np.array([[0, 0], [1, 1]]).astype(float)  # Two points for linear by default
        self.scatter = DraggableScatterPlotItem(
            control_points, size=10, brush=pg.mkBrush(255, 0, 0), pen=pg.mkPen(None)
        )
        self.scatter.sigPlotChanged.connect(self.update_curve)

        self.plot_widget.addItem(self.scatter)
        
        self.update_distribution(0)
    
    def update_distribution(self, index):
        self.scatter.is_spline = False

        if index == 0:  # linear
            self.scatter.control_points = np.array([[0.0, 0.0], [1.0, 1.0]])
        elif index == 1:  # quadratic
            self.scatter.control_points = np.array([[0, 0], [0.5, 0.25], [1, 1]])
        elif index == 2: # spline
            self.scatter.control_points = np.array([[0.0, 0.0], [1.0, 1.0]])
            self.scatter.is_spline = True

        self.scatter.setData(pos=self.scatter.control_points)
        #self.update_curve() # not actually necessary

    def get_distribution(self, x_dist):
        x = self.scatter.control_points[:, 0]
        y = self.scatter.control_points[:, 1]

        if self.scatter.is_spline:
            idx = np.argsort(x)
            spline = CubicSpline(x[idx], y[idx])
            y_dist = spline(x_dist)
        elif len(x) == 2:
            b = (y[1] - y[0]) / (x[1] - x[0])
            a = y[0] - b * x[0]
            y_dist = a + b * x_dist
        elif len(x) == 3:
            coefficients = fit_quadratic(x, y)
            y_dist = coefficients[0] * x_dist**2 + coefficients[1] * x_dist + coefficients[2]
        else:
            y_dist = np.zeros_like(x_dist)

        return y_dist
    
    def update_curve(self):

        x_plot = np.linspace(0, 1, 100)
        y_plot = self.get_distribution(x_plot)
        
        self.plot_widget.clear()
        self.plot_widget.addItem(self.scatter)
        self.plot_widget.plot(x_plot, y_plot, pen='b')

        self.new_dist.emit()

