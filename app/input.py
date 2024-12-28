
from PyQt6.QtWidgets import QTableWidgetItem, QWidget, QVBoxLayout, QLineEdit, QPushButton
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6 import QtGui

import numpy as np

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

        if None in prop.values():
            return None

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
            InputVar(r"$\rho$", "[kg/m3]", "Air density", float),
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

        if None in oper.values():
            return None

        return oper

    def set_values(self, oper):
        self.vars[0].value = oper['rho']
        self.vars[1].value = oper['c0']
        self.vars[2].value = oper['pref']
        self.vars[3].value = oper['V']

        super().set_values()


class InputWidget(QWidget):
    new_prop = pyqtSignal()
    new_oper = pyqtSignal()

    def __init__(self, parent):
        super().__init__(parent)

        self.prop_table = PropInputTable(self)
        self.oper_table = OperInputTable(self)

        self.prop_table.new_prop.connect(self.new_prop.emit)
        self.oper_table.new_oper.connect(self.new_oper.emit)

        self.layout = QVBoxLayout()
        self.prop_path = QLineEdit()
        self.set_prop_btn = QPushButton("Load propeller")

        self.layout.addWidget(self.prop_path)
        self.layout.addWidget(self.set_prop_btn)

        self.layout.addWidget(self.prop_table)
        self.layout.addWidget(self.oper_table)

        self.setMinimumWidth(400)

        self.setLayout(self.layout)

