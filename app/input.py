
from PyQt6.QtWidgets import QTableWidgetItem
from PyQt6.QtCore import Qt
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
        self.setMinimumWidth(300)

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
        
        try:
            v.dtype(self.item(row, col).text())
        except ValueError:
            self.item(row, col).setBackground(QtGui.QColor(255, 0, 0))
            self.clearSelection()
        else:
            self.item(row, col).setBackground(QtGui.QColor(255, 255, 255))
            v.value = v.dtype(self.item(row, col).text())
            self.item(row, col).setText(str(v.value))


class PropInputTable(InputTable):
    def __init__(self, parent):
        vars = [
            InputVar("$B$", "[-]", "Blade number", int),
            InputVar("$r_t$", "[m]", "Blade tip radius", float)
        ]
        super().__init__(vars, parent)

class OpInputTable(InputTable):
    def __init__(self, parent):
        vars = [
            InputVar("$U$", "[m/s]", "Rotor speed", float),
            InputVar("$V$", "[m/s]", "Wind speed", float)
        ]
        super().__init__(vars, parent)
