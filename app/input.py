
from PyQt6.QtWidgets import QTableWidgetItem, QWidget, QVBoxLayout, QLineEdit, QPushButton, QFileDialog, QDialog, QDialogButtonBox, QMessageBox
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6 import QtGui

import pyqtgraph as pg

import numpy as np
import json

from table import TexQTableWidget

class InputVar():
    def __init__(self, symbol, unit, description, dtype = float):
        self.symbol = symbol
        self.unit = unit
        self.description = description
        self.dtype = dtype
        self.value = None

class InputTable(TexQTableWidget):
    def __init__(self, vars, parent):
        super().__init__(parent)

        self.vars = vars

        self.setColumnCount(3)
        self.setRowCount(len(self.vars))
        
        self.setHorizontalHeaderLabels(["Value", "Units", "Description"])

        self.assemble_table()

        self.cellChanged.connect(self.on_cell_changed)

    def assemble_table(self):
        row_names = []
        for i,v in enumerate(self.vars):
            row_names.append(v.symbol)
            self.setItem(
                i, 0, QTableWidgetItem()
            )
            self.setItem(
                i, 1, QTableWidgetItem(v.unit)
            )
            self.setItem(
                i, 2, QTableWidgetItem(v.description)
            )
        if len(row_names) > 0:
            self.setVerticalHeaderLabels(row_names)
    
    def on_cell_changed(self, row, col):
        if col != 0:
            return
        
        try:
            v = self.vars[row]
        except IndexError:
            return
        
        self.cellChanged.disconnect(self.on_cell_changed)

        try:
            v.dtype(self.item(row, col).text())
        except ValueError:
            self.item(row, col).setBackground(QtGui.QColor(255, 0, 0))
            self.clearSelection()
        except AttributeError:
            print(self.item(row, col))
        else:
            v.value = v.dtype(self.item(row, col).text())
            self.item(row, col).setBackground(QtGui.QColor(255, 255, 255))
            self.item(row, col).setText(str(v.value))

        self.cellChanged.connect(self.on_cell_changed)
    
    def set_values(self):
        self.cellChanged.disconnect(self.on_cell_changed)

        for i,v in enumerate(self.vars):
            self.item(i, 0).setText(str(v.value))

        self.cellChanged.connect(self.on_cell_changed)


class PropInputTable(InputTable):
    new_prop = pyqtSignal()

    def __init__(self, parent):
        vars = [
            InputVar("$B$", "[-]", "Blade number", int),
            InputVar("$r_t$", "[m]", "Blade tip radius", float),
            InputVar("$r_h$", "[m]", "Blade hub radius", float),
            InputVar("$n_r$", "[-]", "Radial sections", int),
            InputVar("$n_x$", "[-]", "Chordwise elements per section", int)

        ]
        super().__init__(vars, parent)
    
    def on_cell_changed(self, row, col):
        super().on_cell_changed(row, col)
        self.new_prop.emit()

    
    def parse_values(self):
        prop = {}
        prop['B'] = self.vars[0].value
        prop['rt'] = self.vars[1].value
        prop['rh'] = self.vars[2].value
        prop['nr'] = self.vars[3].value # radial sections
        prop['nx'] = self.vars[4].value # chordwise elements per section

        xc_pts = np.linspace(-0.5,0.5,prop['nx']+1)
        rarr_pts = np.linspace(prop['rh'], prop['rt'], prop['nr']+1)
        prop['xc']  = (xc_pts[1:] + xc_pts[:-1]) / 2
        prop['r0_rt'] = (rarr_pts[1:] + rarr_pts[:-1]) / 2
        prop['dz'] = np.diff(rarr_pts)

        return prop
    
    def set_values(self, prop):
        self.vars[0].value = prop['B']
        self.vars[1].value = prop['rt']
        self.vars[2].value = prop['rh']
        self.vars[3].value = prop['nr']
        self.vars[4].value = prop['nx']

        super().set_values()

class OperInputTable(InputTable):
    new_oper = pyqtSignal()

    def __init__(self, parent):
        vars = [
            InputVar(r"$\rho_0$", "[kg/m3]", "Air density", float),
            InputVar(r"$c_0$", "[m/s]", "Speed of sound", float),
            InputVar(r"$p_\text{ref}$", "[Pa]", "Reference pressure for SPL", float),
            InputVar(r"$V$", "[m/s]", "Free stream velocity", float),
        ]
        super().__init__(vars, parent)
    
    def on_cell_changed(self, row, col):
        super().on_cell_changed(row, col)
        self.new_oper.emit()

    
    def parse_values(self):
        oper = {}
        oper['rho'] = self.vars[0].value
        oper['c0'] = self.vars[1].value
        oper['pref'] = self.vars[2].value
        oper['V'] = self.vars[3].value

        return oper

    def set_values(self, oper):
        self.vars[0].value = oper['rho']
        self.vars[1].value = oper['c0']
        self.vars[2].value = oper['pref']
        self.vars[3].value = oper['V']

        super().set_values()


class AirfoilPlotDialog(QDialog):
    def __init__(self, parent=None, airfoil_data=None):
        super().__init__(parent)
        self.setWindowTitle("Airfoil Plot")
        self.airfoil_data = airfoil_data

        layout = QVBoxLayout()
        
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

        self.setLayout(layout)

        if self.airfoil_data is not None:
            self.plot_airfoil()

    def plot_airfoil(self):
        x, z = self.airfoil_data[:, 0], self.airfoil_data[:, 1]
        self.plot_widget.clear()
        self.plot_widget.plot(x, z, pen=pg.mkPen(color='b', width=2), name="Airfoil Shape")
        self.plot_widget.setLabel("left", "z-coordinate")
        self.plot_widget.setLabel("bottom", "x-coordinate")
        self.plot_widget.setTitle("Airfoil Shape")
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.showGrid(x=True, y=True)

class InputWidget(QWidget):
    new_prop = pyqtSignal()
    new_oper = pyqtSignal()

    def __init__(self, parent):
        super().__init__(parent)

        self.prop_table = PropInputTable(self)
        self.oper_table = OperInputTable(self)

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

        self.setMinimumWidth(400)

        self.setLayout(self.layout)
    
    def connect_signals(self):

        self.prop_table.new_prop.connect(self.on_new_prop)
        self.oper_table.new_oper.connect(self.on_new_oper)

        self.load_prop_btn.clicked.connect(self.load_prop_from_click)
        self.load_foil_btn.clicked.connect(self.load_foil_from_click)

    def on_new_prop(self):
        self.prop = self.prop_table.parse_values()
        self.new_prop.emit()
    
    def on_new_oper(self):
        self.oper = self.oper_table.parse_values()
        self.new_oper.emit()

    def load_prop_from_click(self):
        path = QFileDialog.getOpenFileName(self, "Select propeller file", filter="Propeller files (*.prop)")[0]
        if path:
            self.prop_path.setText(path)

    def load_foil_from_click(self):
        path = QFileDialog.getOpenFileName(self, "Select airfoil file", filter="Airfoil files (*.surf)")[0]
        if not path:
            return
        
        try:
            airfoil_data = np.loadtxt(path)
        except:
            QMessageBox.critical(self, "Error", "Failed to load airfoil data from file.")
            return
        
        dialog = AirfoilPlotDialog(self, airfoil_data)
        dialog.exec()

        if dialog.result():
            self.foil_path.setText(path)
            self.airfoil_data = airfoil_data
            self.new_prop.emit()

    
    def load_oper_from_file(self, filename):
        try:
            with open(filename, 'r') as f:
                av = json.load(f)
        except FileNotFoundError:
            return False
        self.oper = av['oper']
        self.oper_table.set_values(self.oper)
        return True

    def save_oper_to_file(self, filename):
        av = {
            'oper': self.oper
        }
        try:
            with open(filename, 'w') as f:
                json.dump(av, f)
        except FileNotFoundError:
            print("Error saving to directory")
            return

        
