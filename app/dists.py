import bem as bem
import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QStandardItem, QStandardItemModel
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from routines import calc_distribution
from scipy.interpolate import CubicSpline


class DraggableScatterPlotItem(pg.ScatterPlotItem):
    def __init__(self, control_points, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_points = control_points
        self.dragged_point_index = None
        self.is_spline = False
        self.finished_dragging = False
        self.grid_snap = False
        self.distribution_index = 0

    def mouseDragEvent(self, ev):

        if ev.button() != pg.QtCore.Qt.MouseButton.LeftButton:
            ev.ignore()
            return

        if not self.isVisible():
            ev.ignore()
            return

        mouspos = np.array([ev.pos().x(), ev.pos().y()])
        viewrange = np.array([self.viewRect().width(), self.viewRect().height()])

        if self.grid_snap:
            # get snap size as order of magnitude of the range

            snap_size = 10 ** (np.round(np.log10(viewrange)) - 1)
            snappos = np.round(mouspos / snap_size) * snap_size
        else:
            snappos = mouspos

        if ev.isStart():
            distances = [
                np.linalg.norm((mouspos - p) / viewrange) for p in self.control_points
            ]
            point_threshold = 0.05
            if (
                np.min(distances) > point_threshold
            ):  # far from existing points try to add a new one
                if not self.is_spline:  # only spline can add points
                    ev.ignore()
                    return
                # compute distribution at mouse position
                ydist = calc_distribution(
                    self.distribution_index, self.control_points, np.array([mouspos[0]])
                )
                line_threshold = 0.05
                if (
                    np.abs(ydist - mouspos[1]) / viewrange[1] > line_threshold
                ):  # clicked far from distribution ignore
                    ev.ignore()
                    return

                self.control_points = np.vstack([self.control_points, snappos])
                self.dragged_point_index = self.control_points.shape[0] - 1
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
                self.control_points[self.dragged_point_index] = snappos
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

        mouspos = np.array([ev.pos().x(), ev.pos().y()])
        viewrange = np.array([self.viewRect().width(), self.viewRect().height()])
        distances = [
            np.linalg.norm((mouspos - p) / viewrange) for p in self.control_points
        ]
        point_threshold = 0.05
        if (
            np.min(distances) < point_threshold
        ):  # far from existing points try to add a new one
            self.control_points = np.delete(
                self.control_points, np.argmin(distances), axis=0
            )
            self.setData(pos=self.control_points)
            self.finished_dragging = (
                True  # not strictly true but we want to update the curve
            )
            self.sigPlotChanged.emit(self)


class DistributionPlotWidget(QWidget):
    new_dist = pyqtSignal(bool)

    def __init__(
        self, parent=None, default_ctrl_pts=None, title=None, xlabel=None, ylabel=None
    ):
        super().__init__(parent)

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.dist_type = QComboBox()
        self.dist_model = QStandardItemModel(self.dist_type)
        # self.dist_type.addItems(["Linear", "Quadratic", "Spline", "Betz"])
        self.dist_model.appendRow(QStandardItem("Linear"))
        self.dist_model.appendRow(QStandardItem("Quadratic"))
        self.dist_model.appendRow(QStandardItem("Spline"))
        custom_item = QStandardItem("Custom")
        custom_item.setFlags(custom_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        self.dist_model.appendRow(custom_item)
        self.dist_model.appendRow(QStandardItem("Inverse"))
        self.dist_model.appendRow(QStandardItem("arctan(1/r)"))
        self.dist_type.setModel(self.dist_model)
        self.dist_type.currentIndexChanged.connect(self.reset_distribution)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setXRange(0, 1)
        self.plot_widget.setYRange(0, 1)
        self.plot_widget.setLabel("left", "Value")
        self.plot_widget.setLabel("bottom", "Radius")

        if title is not None:
            self.plot_widget.setTitle(title)
        if xlabel is not None:
            self.plot_widget.setLabel("bottom", xlabel)
        if ylabel is not None:
            self.plot_widget.setLabel("left", ylabel)
        self.plot_widget.showGrid(x=True, y=True)

        layout.addWidget(self.plot_widget)
        layout.addWidget(self.dist_type)

        # initially linear
        if default_ctrl_pts is None:
            default_ctrl_pts = np.array([[0.0, 0.1], [1.0, 0.1], [0.5, 0.2]])
        elif default_ctrl_pts.shape[0] < 3:
            raise ValueError("Control points must have at least 3 points")

        self.default_ctrl_pts = default_ctrl_pts
        self.scatter = DraggableScatterPlotItem(
            default_ctrl_pts, size=10, brush=pg.mkBrush(255, 0, 0), pen=pg.mkPen(None)
        )
        self.scatter.sigPlotChanged.connect(self.update_curve)

        self.plot_widget.addItem(self.scatter)

        self.reset_distribution(0)

        self.xb = np.linspace(0, 1, 100)
        self.yb = np.zeros_like(self.xb)

    def reset_distribution(self, index):
        self.scatter.is_spline = False
        self.scatter.finished_dragging = True
        self.scatter.distribution_index = index
        self.scatter.show()
        self.set_distribution(index, *self.default_ctrl_pts)

    def get_distribution(self, x_dist):
        dist_index = self.dist_type.currentIndex()
        if dist_index == 3:
            if x_dist.shape != self.xb.shape:
                self.yb = np.interp(x_dist, self.xb, self.yb)
                self.xb = x_dist
            return self.yb
        else:
            return calc_distribution(dist_index, self.scatter.control_points, x_dist)

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
        self.plot_widget.plot(x_plot, y_plot, pen="b")

        if self.scatter.finished_dragging:
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

        self.new_dist.emit(self.scatter.finished_dragging)
        self.scatter.finished_dragging = False

    def set_distribution(self, index, *args):
        self.scatter.is_spline = False
        self.scatter.show()
        npargs = np.array(args).reshape(-1, 2)

        self.dist_type.currentIndexChanged.disconnect(self.reset_distribution)

        if index == 0:
            self.dist_type.setCurrentIndex(0)
            self.scatter.control_points = npargs[0:2]
        elif index == 1:
            self.dist_type.setCurrentIndex(1)
            self.scatter.control_points = npargs[0:3]
        elif index == 2:
            self.dist_type.setCurrentIndex(2)
            self.scatter.control_points = npargs[0:3]
            self.scatter.is_spline = True
        elif index == 3:
            self.dist_type.setCurrentIndex(3)
            self.scatter.control_points = np.zeros((0, 2))
            self.xb = npargs[0]
            self.yb = npargs[1]
            self.scatter.hide()
        elif index == 4:
            self.dist_type.setCurrentIndex(4)
            self.scatter.control_points = npargs[0:2]
        elif index == 5:
            self.dist_type.setCurrentIndex(5)
            self.scatter.control_points = npargs[0:1]

        self.scatter.setData(pos=self.scatter.control_points)
        self.dist_type.currentIndexChanged.connect(self.reset_distribution)

        miny = np.min(self.scatter.control_points[:, 1])
        maxy = np.max(self.scatter.control_points[:, 1])
        padding = 0.1 * (maxy - miny) if maxy > miny else 0.1
        self.plot_widget.setYRange(miny - padding, maxy + padding)
        self.update_curve()


class DistributionsWidget(QWidget):
    new_dist = pyqtSignal(bool)
    new_prop = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)

        self.betz_button = QPushButton("Betz Optimised")
        self.betz_button.clicked.connect(self.betz_optimised)
        layout.addWidget(self.betz_button)

        self.avs = None
        self.distypes = ["linear", "quadratic", "spline", "custom", "inverse", "arctan"]

        self.chord_plot = DistributionPlotWidget(
            self, title="Chord", ylabel="Chord [m]"
        )
        self.twist_plot = DistributionPlotWidget(
            self,
            title="Twist",
            ylabel="Twist [deg]",
            default_ctrl_pts=np.array([[0, 20], [0.5, 15], [1, 15]]),
        )
        self.sweep_plot = DistributionPlotWidget(
            self,
            title="Sweep",
            ylabel="Sweep [deg]",
            default_ctrl_pts=np.array([[0, 0], [0.5, 5], [1, 20]]),
        )

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

    def on_new_dist(self, finished_dragging=False):
        self.new_dist.emit(finished_dragging)

    def update_avs(self, avs):

        try:
            avs.prop["r0_rt"]
        except KeyError:
            return

        # high resolution distributions
        avs.prop["c"] = (
            self.chord_plot.get_distribution(avs.prop["r0_rt"]) * avs.prop["c75"]
        )
        avs.prop["twist"] = (
            self.twist_plot.get_distribution(avs.prop["r0_rt"]) * np.pi / 180
        )
        avs.prop["sweep"] = (
            self.sweep_plot.get_distribution(avs.prop["r0_rt"]) * np.pi / 180
        )

        avs.dist["CTL_c"] = self.chord_plot.scatter.control_points
        avs.dist["CTL_c_type"] = self.distypes[self.chord_plot.dist_type.currentIndex()]
        avs.dist["CTL_twist"] = self.twist_plot.scatter.control_points
        avs.dist["CTL_twist_type"] = self.distypes[
            self.twist_plot.dist_type.currentIndex()
        ]
        avs.dist["CTL_sweep"] = self.sweep_plot.scatter.control_points
        avs.dist["CTL_sweep_type"] = self.distypes[
            self.sweep_plot.dist_type.currentIndex()
        ]

        self.avs = avs

    def betz_optimised(self):
        if self.avs is None:
            return

        if bem.betz_design(self.avs):
            self.set_dists(self.avs)

            bem.betz_off_design(self.avs)

        else:
            QMessageBox.critical(
                self, "Error", "Betz Optimisation Failed", QMessageBox.StandardButton.Ok
            )

    def set_dists(self, avs):
        self.detach_dist_signals()

        chord_dist = self.distypes.index(avs.dist["CTL_c_type"])
        if chord_dist == 3:
            self.chord_plot.set_distribution(
                3, avs.prop["r0_rt"], avs.prop["c"] / avs.prop["c75"]
            )
        else:
            self.chord_plot.set_distribution(chord_dist, avs.dist["CTL_c"])

        twist_dist = self.distypes.index(avs.dist["CTL_twist_type"])
        if twist_dist == 3:
            self.twist_plot.set_distribution(
                3, avs.prop["r0_rt"], avs.prop["twist"] * 180 / np.pi
            )
        else:
            self.twist_plot.set_distribution(twist_dist, avs.dist["CTL_twist"])

        sweep_dist = self.distypes.index(avs.dist["CTL_sweep_type"])
        if sweep_dist == 3:
            self.sweep_plot.set_distribution(
                3, avs.prop["r0_rt"], avs.prop["sweep"] * 180 / np.pi
            )
        else:
            self.sweep_plot.set_distribution(sweep_dist, avs.dist["CTL_sweep"])

        self.attach_dist_signals()
        self.new_dist.emit(True)  # full update please

    def keyPressEvent(self, a0):
        if a0.key() == Qt.Key.Key_Shift:
            self.chord_plot.scatter.grid_snap = True
            self.twist_plot.scatter.grid_snap = True
            self.sweep_plot.scatter.grid_snap = True
        return super().keyPressEvent(a0)

    def keyReleaseEvent(self, a0):
        if a0.key() == Qt.Key.Key_Shift:
            self.chord_plot.scatter.grid_snap = False
            self.twist_plot.scatter.grid_snap = False
            self.sweep_plot.scatter.grid_snap = False
        return super().keyReleaseEvent(a0)
