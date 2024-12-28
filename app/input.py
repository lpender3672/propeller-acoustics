
from PyQt6.QtWidgets import QTableWidgetItem, QWidget, QVBoxLayout
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
        else:
            v.value = v.dtype(self.item(row, col).text())
            self.item(row, col).setBackground(QtGui.QColor(255, 255, 255))
            self.item(row, col).setText(str(v.value))

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

        return oper


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
        self.layout.addWidget(self.prop_table)
        self.layout.addWidget(self.oper_table)

        self.setMinimumWidth(400)

        self.setLayout(self.layout)

