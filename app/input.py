import json
import os
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtGui
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.table import (
    InputTable, TableBox, TableVar
)

from app.routines import (
    interpolate_clcd, load_foil, run_xfoil
)

XFOIL_INSTALLED = True
try:
    from xfoil import XFoil
    from xfoil.model import Airfoil
except ModuleNotFoundError:
    XFOIL_INSTALLED = False
    print("Warning Xfoil not installed - betz.py")


class PropInputTable(InputTable):
    new_prop = pyqtSignal()

    def __init__(self, parent):
        vars = [
            TableVar(r"$B$", "[-]", "Blade number", int),
            TableVar(r"$r_t$", "[m]", "Blade tip radius", float),
            TableVar(r"$r_h$", "[m]", "Blade hub radius", float),
            TableVar(r"$n_r$", "[-]", "Radial sections", int),
            TableVar(r"$n_x$", "[-]", "Chordwise elements per section", int),
            TableBox(["Linear", "Cosine"], "rdist", "Radial distribution"),
            TableVar(r"$c_{75}$", "[m]", "75% Chord", float),
            TableBox(["Single", "Symmetric", "Asymmetric"], "twinB", "Twin blade type"),
            TableVar(r"$D_{th}$", "[m]", "Thead diameter", float),
            TableBox(["CW", "CCW"], "bdir", "Blade direction"),
        ]
        self.keys = [
            "B",
            "rt",
            "rh",
            "nr",
            "nx",
            "rdist",
            "c75",
            "twinB",
            "dthread",
            "bdir",
        ]
        super().__init__(vars, parent)

    def on_cell_changed(self, row, col):
        super().on_cell_changed(row, col)
        self.new_prop.emit()

    def parse_values(self):
        prop = {}
        vals = [v.value for v in self.vars]
        if None in vals:
            return None

        for i, key in enumerate(self.keys):
            prop[key] = vals[i]

        xc_pts = np.linspace(-0.5, 0.5, prop["nx"] + 1)
        if prop["rdist"] == "Linear":
            rarr_pts = np.linspace(prop["rh"], prop["rt"], prop["nr"] + 1)
        elif prop["rdist"] == "Cosine":
            rarr_pts = prop["rh"] + (prop["rt"] - prop["rh"]) / 2 * (
                1 - np.cos(np.linspace(0, np.pi, prop["nr"] + 1))
            )

        prop["xc"] = (xc_pts[1:] + xc_pts[:-1]) / 2
        prop["r0"] = (rarr_pts[1:] + rarr_pts[:-1]) / 2
        prop["r0_rt"] = prop["r0"] / prop["rt"]
        prop["dz"] = np.diff(rarr_pts)

        return prop

    def set_values(self, prop):

        for i, key in enumerate(self.keys):
            self.vars[i].value = prop[key]

        super().set_values()


class OperInputTable(InputTable):
    new_oper = pyqtSignal()

    def __init__(self, parent):
        vars = [
            TableVar(r"$\rho_0$", "[kg/m3]", "Air density", float),
            TableVar(r"$\nu$", "[m2/s]", "Kinematic viscosity", float),
            TableVar(r"$c_0$", "[m/s]", "Speed of sound", float),
            TableVar(r"$p_{ref}$", "[Pa]", "Reference pressure for SPL", float),
            TableVar(r"$V$", "[m/s]", "Free stream velocity", float),
            TableVar(r"$\Omega$", "[RPM]", "Rotational speed", float),
            TableVar(r"$r_{obs}/r_t$", "[-]", "Observer distance", float),
            TableVar(r"$\theta_{obs}$", "[deg]", "Observer angle", float),
        ]
        self.keys = ["rho", "nu", "c0", "pref", "V", "Omega", "r_obs", "theta_obs"]
        super().__init__(vars, parent)

    def on_cell_changed(self, row, col):
        super().on_cell_changed(row, col)
        self.new_oper.emit()

    def parse_values(self):
        oper = {}
        vals = [v.value for v in self.vars]
        if None in vals:
            return None

        for i, key in enumerate(self.keys):
            oper[key] = vals[i]

        oper["Omega"] = oper["Omega"] * 2 * np.pi / 60
        oper["theta_obs"] = oper["theta_obs"] * np.pi / 180

        return oper

    def set_values(self, oper):

        for i, key in enumerate(self.keys):
            self.vars[i].value = oper[key]

        self.vars[5].value = oper["Omega"] * 60 / (2 * np.pi)
        self.vars[7].value = oper["theta_obs"] * 180 / np.pi

        super().set_values()


class AirfoilPlotDialog(QDialog):
    def __init__(self, parent=None, airfoil_data=None, xfoil_data=None):
        super().__init__(parent)
        self.setWindowTitle("Airfoil Plot")
        self.xfoil_data = xfoil_data
        self.airfoil_data = airfoil_data

        layout = QVBoxLayout()

        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)
        self.swap_plot_btn = QPushButton("Swap Plot")
        self.swap_plot_btn.clicked.connect(self.plot_airfoil)
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.swap_plot_btn)
        layout.addWidget(self.button_box)

        self.setLayout(layout)

        if self.airfoil_data is not None:
            self.plot_airfoil()

    def plot_airfoil(self):
        self.swap_plot_btn.clicked.disconnect(self.plot_airfoil)
        self.swap_plot_btn.clicked.connect(self.plot_polar)

        x, z = self.airfoil_data[:, 0], self.airfoil_data[:, 1]
        self.plot_widget.clear()
        self.plot_widget.plot(
            x, z, pen=pg.mkPen(color="b", width=2), name="Airfoil Shape"
        )
        self.plot_widget.setLabel("left", "z-coordinate")
        self.plot_widget.setLabel("bottom", "x-coordinate")
        self.plot_widget.setTitle("Airfoil Shape")
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.showGrid(x=True, y=True)

    def plot_polar(self):

        if self.xfoil_data.shape[1] < 4:
            return

        self.swap_plot_btn.clicked.disconnect(self.plot_polar)
        self.swap_plot_btn.clicked.connect(self.plot_airfoil)

        alpha = self.xfoil_data[:, 0]
        Re = self.xfoil_data[:, 1]
        Cl = self.xfoil_data[:, 2]
        Cd = self.xfoil_data[:, 3]

        x = np.linspace(np.min(alpha) * np.pi / 180, np.pi / 2, 100)
        y1, y2 = interpolate_clcd(self.xfoil_data, x, 5e5)

        self.plot_widget.clear()
        self.plot_widget.plot(
            x * 180 / np.pi, y1, pen=pg.mkPen(color="b", width=2), name="Cl"
        )
        self.plot_widget.plot(
            x * 180 / np.pi, y2, pen=pg.mkPen(color="r", width=2), name="Cd"
        )
        self.plot_widget.setLabel("left", "Coefficient")
        self.plot_widget.setLabel("bottom", "Angle of Attack")
        self.plot_widget.setTitle("Polar")
        self.plot_widget.setAspectLocked(False)
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.addLegend()


class XFoilThread(QThread):

    finished = pyqtSignal(np.ndarray)

    def __init__(self, airfoil_data):
        super().__init__()
        self.airfoil_data = airfoil_data

    def run(self):
        xfoil_data = run_xfoil(self.airfoil_data)
        self.finished.emit(xfoil_data)


class InputWidget(QWidget):
    new_prop = pyqtSignal()
    new_oper = pyqtSignal()
    new_prop_from_file = pyqtSignal()

    def __init__(self, parent):
        super().__init__(parent)

        self.prop_table = PropInputTable(self)
        self.oper_table = OperInputTable(self)

        self.prop = {}
        self.dist = {}
        self.prop_defined = False
        self.oper_defined = False

        self.assemble_widgets()
        self.connect_signals()

    def assemble_widgets(self):

        self.layout = QVBoxLayout()
        self.prop_path = QLineEdit()
        self.prop_path.setReadOnly(True)
        self.load_prop_btn = QPushButton("Load propeller")
        self.save_prop_btn = QPushButton("Save propeller")

        self.foil_path = QLineEdit()
        self.foil_path.setReadOnly(True)
        self.load_foil_btn = QPushButton("Load airfoil")
        self.airfoil_data = None

        self.layout.addWidget(self.prop_path)
        self.layout.addWidget(self.load_prop_btn)
        self.layout.addWidget(self.save_prop_btn)

        self.layout.addWidget(self.prop_table)

        self.layout.addWidget(self.foil_path)
        self.layout.addWidget(self.load_foil_btn)

        self.layout.addWidget(self.oper_table)

        self.setLayout(self.layout)

    def connect_signals(self):

        self.prop_table.new_prop.connect(self.on_new_prop)
        self.oper_table.new_oper.connect(self.on_new_oper)

        self.load_prop_btn.clicked.connect(self.load_prop_from_click)
        self.load_foil_btn.clicked.connect(self.load_foil_from_click)

    def on_new_prop(self):
        prop = self.prop_table.parse_values()
        if prop is not None:
            self.prop_defined = True
            self.prop.update(prop)
            self.new_prop.emit()

    def on_new_oper(self):
        oper = self.oper_table.parse_values()
        if oper is not None:
            self.oper_defined = True
            self.oper.update(oper)
            self.new_oper.emit()

    def load_prop_from_click(self):
        path = QFileDialog.getOpenFileName(
            self,
            "Select propeller file",
            "app/props",
            filter="Propeller files (*.prop)",
        )[0]
        if path:
            self.prop_path.setText(path)
        else:
            return

        try:
            with open(path, "r") as f:
                indata = json.load(f)
        except BaseException:
            QMessageBox.critical(
                self, "Error", "Failed to load propeller data from file."
            )
            return

        try:
            filepath = indata["prop"]["foil_path"]
            airfoil_data = load_foil(filepath)
        except (KeyError, IndexError):
            QMessageBox.critical(
                self, "Error", "Failed to load airfoil data from file for propeller."
            )
            return
        except FileNotFoundError:
            # search for file name in current directory
            current_dir = Path(os.getcwd())
            filepath = Path(filepath)
            # call some kind of recursive search function
            for file in current_dir.rglob("*.surf"):
                if file.name == filepath.name:
                    filepath = str(file)
                    indata["prop"]["foil_path"] = filepath
                    break
            else:
                QMessageBox.critical(
                    self, "Error", "Failed to find airfoil file in current directory."
                )
                return
            airfoil_data = load_foil(filepath)

        self.foil_path.setText(filepath)
        self.airfoil_data = airfoil_data
        self.prop = indata["prop"]
        for key, item in self.prop.items():
            if isinstance(item, list):
                self.prop[key] = np.array(item)
        self.dist = indata["dist"]
        for key, item in self.dist.items():
            if isinstance(item, list):
                self.dist[key] = np.array(item)

        self.xfoil_thread = XFoilThread(airfoil_data)
        self.xfoil_thread.finished.connect(
            self.xfoil_thread_finished_after_load_prop_clicked
        )
        self.xfoil_thread.start()

    def xfoil_thread_finished_after_load_prop_clicked(self, xfoil_data):
        self.xfoil_thread.finished.disconnect(
            self.xfoil_thread_finished_after_load_prop_clicked
        )
        self.xfoil_data = xfoil_data
        self.prop_defined = True
        self.prop_table.set_values(self.prop)
        self.new_prop_from_file.emit()

    def load_foil_from_click(self):
        path = QFileDialog.getOpenFileName(
            self, "Select airfoil file", "app/foils", filter="Airfoil files (*.surf)"
        )[0]
        if path:
            self.foil_path.setText(path)
        else:
            return

        try:
            airfoil_data = load_foil(path)
        except BaseException:
            QMessageBox.critical(
                self, "Error", "Failed to load airfoil data from file."
            )
            return

        self.foil_path.setText(path)
        self.prop["foil_path"] = path
        self.airfoil_data = airfoil_data

        self.xfoil_thread = XFoilThread(airfoil_data)
        self.xfoil_thread.finished.connect(
            self.xfoil_thread_finished_after_load_foil_clicked
        )
        self.xfoil_thread.start()

    def xfoil_thread_finished_after_load_foil_clicked(self, xfoil_data):

        self.xfoil_data = xfoil_data

        dialog = AirfoilPlotDialog(self, self.airfoil_data, self.xfoil_data)
        dialog.exec()

        if dialog.result():
            self.on_new_prop()

    def save_prop_to_file(self, avs):
        path = QFileDialog.getSaveFileName(
            self, "Select propeller file", filter="Propeller files (*.prop)"
        )[0]
        if not path:
            return

        parsed_prop = {}
        parsed_prop["foil_path"] = self.foil_path.text()

        for key, item in avs.prop.items():
            if isinstance(item, np.ndarray):
                parsed_prop[key] = item.tolist()
            else:
                parsed_prop[key] = item

        parsed_dist = {}
        for key, item in avs.dist.items():
            if isinstance(item, np.ndarray):
                parsed_dist[key] = item.tolist()
            else:
                parsed_dist[key] = item

        out = {"prop": parsed_prop, "dist": parsed_dist}
        try:
            with open(path, "w") as f:
                json.dump(out, f, indent=4)
        except BaseException:
            QMessageBox.critical(
                self, "Error", "Failed to save propeller data to file."
            )
            return

    def load_oper_from_file(self, filename):
        try:
            with open(filename, "r") as f:
                av = json.load(f)
        except FileNotFoundError:
            return False
        self.oper = av["oper"]
        self.oper_table.set_values(self.oper)
        self.on_new_oper()
        return True

    def save_oper_to_file(self, filename):
        av = {"oper": self.oper}
        try:
            with open(filename, "w") as f:
                json.dump(av, f, indent=4)
        except FileNotFoundError:
            print("Error saving to directory")
            return
