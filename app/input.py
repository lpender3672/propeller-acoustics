
from PyQt6.QtWidgets import QTableWidgetItem, QWidget, QVBoxLayout, QLineEdit, QPushButton, QFileDialog, QDialog, QDialogButtonBox, QMessageBox, QComboBox
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6 import QtGui

import pyqtgraph as pg

import numpy as np
import json
from pathlib import Path
import os


XFOIL_INSTALLED = True
try: 
    from xfoil import XFoil
    from xfoil.model import Airfoil
except ModuleNotFoundError:
    XFOIL_INSTALLED = False
    print("Warning Xfoil not installed - betz.py")


from table import TexQTableWidget

class InputVar():
    def __init__(self, symbol, unit, description, dtype = float):
        self.symbol = symbol
        self.unit = unit
        self.description = description
        self.dtype = dtype
        self.value = None

class InputBox(QComboBox):
    def __init__(self, items, symbol, description):
        super().__init__()
        self.addItems(items)

        self.symbol = symbol
        self.description = description
        self.unit = ""
        self.dtype = str
        self.value = None


def foil_data(airfoil_data, alpha, Re):
    
    xf = XFoil()
    xf.airfoil = Airfoil(
        airfoil_data[:,0],
        airfoil_data[:,1]
        )
    xf.Re = Re
    xf.M = 0.0
    xf.max_iter = 100
    xf.verbose = False

    if isinstance(alpha, float):
        cls = np.zeros(1)
        cds = np.zeros(1)
        alpha = [alpha]
    else:
        cls = np.zeros(len(alpha))
        cds = np.zeros(len(alpha))

    for i, a in enumerate(alpha):
        out = xf.a(a)
        print(out)
        cl, cd, _, _ = out
        cls[i] = cl
        cds[i] = cd

    return cls, cds

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
            if isinstance(v, InputVar):
                self.setItem(
                    i, 0, QTableWidgetItem()
                )
            elif isinstance(v, InputBox):
                self.setCellWidget(
                    i, 0, v
                )
                v.currentIndexChanged.connect(
                        lambda index, r=i, c=0: self.on_cell_changed(r, c, index)
                )
            self.setItem(
                i, 1, QTableWidgetItem(v.unit)
            )
            self.setItem(
                i, 2, QTableWidgetItem(v.description)
            )
        if len(row_names) > 0:
            self.setVerticalHeaderLabels(row_names)
    
    def on_cell_changed(self, row, col, index = None):
        if col != 0:
            return
        
        try:
            v = self.vars[row]
        except IndexError:
            return
        
        self.cellChanged.disconnect(self.on_cell_changed)

        if isinstance(v, InputBox):
            v.value = v.currentText()
        
        elif isinstance(v, InputVar):
            try:
                v.dtype(self.item(row, col).text())
            except ValueError:
                self.item(row, col).setBackground(QtGui.QColor(255, 0, 0))
                self.clearSelection()
            except AttributeError:
                print(self.item(row, col))
            else:
                v.value = v.dtype(self.item(row, col).text())
                # get color of theme
                colour = self.palette().color(QtGui.QPalette.ColorRole.Base)
                self.item(row, col).setBackground(colour)
                self.item(row, col).setText(str(v.value))

        self.cellChanged.connect(self.on_cell_changed)
    
    def set_values(self):
        self.cellChanged.disconnect(self.on_cell_changed)

        for i,v in enumerate(self.vars):
            if isinstance(v, InputVar):
                self.item(i, 0).setText(str(v.value))
            elif isinstance(v, InputBox):
                # set index of combobox
                self.cellWidget(i, 0).setCurrentText(v.value)

        self.cellChanged.connect(self.on_cell_changed)


class PropInputTable(InputTable):
    new_prop = pyqtSignal()

    def __init__(self, parent):
        vars = [
            InputVar(r"$B$", "[-]", "Blade number", int),
            InputVar(r"$r_t$", "[m]", "Blade tip radius", float),
            InputVar(r"$r_h$", "[m]", "Blade hub radius", float),
            InputVar(r"$n_r$", "[-]", "Radial sections", int),
            InputVar(r"$n_x$", "[-]", "Chordwise elements per section", int),
            InputBox(["Linear", "Cosine"], "rdist", "Radial distribution"),
            InputVar(r"$c_{75}$", "[m]", "75% Chord", float),
        ]
        self.keys = ["B", "rt", "rh", "nr", "nx", "rdist", "c75"]
        super().__init__(vars, parent)
    
    def on_cell_changed(self, row, col, index=None):
        super().on_cell_changed(row, col, index)
        self.new_prop.emit()

    
    def parse_values(self):
        prop = {}
        vals = [self.vars[i].value for i in range(7)]
        if None in vals:
            return None

        for i, key in enumerate(self.keys):
            prop[key] = vals[i]
        
        xc_pts = np.linspace(-0.5,0.5,prop['nx']+1)
        if prop['rdist'] == "Linear":
            rarr_pts = np.linspace(prop['rh'], prop['rt'], prop['nr']+1)
        elif prop['rdist'] == "Cosine":
            rarr_pts = prop['rh'] + (prop['rt']-prop['rh'])/2 * (1 - np.cos(np.linspace(0,np.pi,prop['nr']+1)))

        prop['xc']  = (xc_pts[1:] + xc_pts[:-1]) / 2
        prop['r0'] = (rarr_pts[1:] + rarr_pts[:-1]) / 2
        prop['r0_rt'] = prop['r0'] / prop['rt']
        prop['dz'] = np.diff(rarr_pts)

        return prop
    
    def set_values(self, prop):

        for i, key in enumerate(self.keys):
            self.vars[i].value = prop[key]
                
        super().set_values()

class OperInputTable(InputTable):
    new_oper = pyqtSignal()

    def __init__(self, parent):
        vars = [
            InputVar(r"$\rho_0$", "[kg/m3]", "Air density", float),
            InputVar(r"$\nu$", "[m2/s]", "Kinematic viscosity", float),
            InputVar(r"$c_0$", "[m/s]", "Speed of sound", float),
            InputVar(r"$p_{ref}$", "[Pa]", "Reference pressure for SPL", float),
            InputVar(r"$V$", "[m/s]", "Free stream velocity", float),
            InputVar(r"$\Omega$", "[RPM]", "Rotational speed", float),
        ]
        self.keys = ["rho", "nu", "c0", "pref", "V", "Omega"]
        super().__init__(vars, parent)
    
    def on_cell_changed(self, row, col, index=None):
        super().on_cell_changed(row, col, index)
        self.new_oper.emit()

    
    def parse_values(self):
        oper = {}
        vals = [self.vars[i].value for i in range(6)]
        if None in vals:
            return None
        
        for i, key in enumerate(self.keys):
            oper[key] = vals[i]

        oper['Omega'] = oper['Omega'] * 2*np.pi / 60

        return oper

    def set_values(self, oper):

        for i, key in enumerate(self.keys):
            self.vars[i].value = oper[key]
        
        self.vars[5].value = oper['Omega'] * 60/(2*np.pi)

        super().set_values()


class AirfoilPlotDialog(QDialog):
    def __init__(self, parent=None, airfoil_data=None):
        super().__init__(parent)
        self.setWindowTitle("Airfoil Plot")
        self.airfoil_data = airfoil_data

        layout = QVBoxLayout()
        
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)
        self.swap_plot_btn = QPushButton("Swap Plot")
        self.swap_plot_btn.clicked.connect(self.plot_airfoil)
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
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
        self.plot_widget.plot(x, z, pen=pg.mkPen(color='b', width=2), name="Airfoil Shape")
        self.plot_widget.setLabel("left", "z-coordinate")
        self.plot_widget.setLabel("bottom", "x-coordinate")
        self.plot_widget.setTitle("Airfoil Shape")
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.showGrid(x=True, y=True)
    
    def plot_polar(self):

        if self.airfoil_data.shape[1] < 4:
            return
        
        self.swap_plot_btn.clicked.disconnect(self.plot_polar)
        self.swap_plot_btn.clicked.connect(self.plot_airfoil)
        
        alpha = self.airfoil_data[:, 2]
        Cl = self.airfoil_data[:, 3]
        Cd = self.airfoil_data[:, 4]

        self.plot_widget.clear()
        self.plot_widget.plot(alpha, Cl, pen=pg.mkPen(color='b', width=2), name="Cl")
        self.plot_widget.plot(alpha, Cd, pen=pg.mkPen(color='r', width=2), name="Cd")
        self.plot_widget.setLabel("left", "Coefficient")
        self.plot_widget.setLabel("bottom", "Angle of Attack")
        self.plot_widget.setTitle("Polar")
        self.plot_widget.setAspectLocked(False)
        self.plot_widget.showGrid(x=True, y=True)

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
        path = QFileDialog.getOpenFileName(self, "Select propeller file", "app/props", filter="Propeller files (*.prop)")[0]
        if path:
            self.prop_path.setText(path)
        else:
            return
        
        try:
            with open(path, 'r') as f:
                indata = json.load(f)
        except:
            QMessageBox.critical(self, "Error", "Failed to load propeller data from file.")
            return
        
        try:
            filepath = indata['prop']['foil_path']
            airfoil_data = np.loadtxt(filepath)
        except (KeyError, IndexError):
            QMessageBox.critical(self, "Error", "Failed to load airfoil data from file for propeller.")
            return
        except FileNotFoundError:
            # search for file name in current directory
            current_dir = Path(os.getcwd())
            filepath = Path(filepath)
            # call some kind of recursive search function
            for file in current_dir.rglob('*.surf'):
                if file.name == filepath.name:
                    filepath = str(file)
                    indata['prop']['foil_path'] = filepath
                    break
            else:
                QMessageBox.critical(self, "Error", "Failed to find airfoil file in current directory.")
                return
            airfoil_data = np.loadtxt(filepath)

        
        self.foil_path.setText(filepath)
        self.airfoil_data = self.run_xfoil(airfoil_data)
        
        self.prop = indata['prop']
        for key, item in self.prop.items():
            if isinstance(item, list):
                self.prop[key] = np.array(item)
        self.dist = indata['dist']
        for key, item in self.dist.items():
            if isinstance(item, list):
                self.dist[key] = np.array(item)
        

        self.prop_table.set_values(self.prop)
        self.new_prop_from_file.emit()

    def load_foil_from_click(self):
        path = QFileDialog.getOpenFileName(self, "Select airfoil file", "app/foils", filter="Airfoil files (*.surf)")[0]
        if path:
            self.foil_path.setText(path)
        else:
            return
        
        try:
            airfoil_data = np.loadtxt(path)
        except:
            QMessageBox.critical(self, "Error", "Failed to load airfoil data from file.")
            return
        
        airfoil_data = self.run_xfoil(airfoil_data)
        dialog = AirfoilPlotDialog(self, airfoil_data)
        dialog.exec()

        if dialog.result():
            self.foil_path.setText(path)
            self.airfoil_data = airfoil_data
            self.prop['foil_path'] = path
            self.on_new_prop()

    def save_prop_to_file(self, avs):
        path = QFileDialog.getSaveFileName(self, "Select propeller file", filter="Propeller files (*.prop)")[0]
        if not path:
            return
        
        parsed_prop = {}
        parsed_prop['foil_path'] = self.foil_path.text()

        for key,item in avs.prop.items():
            if isinstance(item, np.ndarray):
                parsed_prop[key] = item.tolist()
            else:
                parsed_prop[key] = item

        parsed_dist = {}
        for key,item in avs.dist.items():
            if isinstance(item, np.ndarray):
                parsed_dist[key] = item.tolist()
            else:
                parsed_dist[key] = item
        
        out = {
            'prop': parsed_prop,
            'dist': parsed_dist
        }
        try:
            with open(path, 'w') as f:
                json.dump(out, f, indent=4)
        except:
            QMessageBox.critical(self, "Error", "Failed to save propeller data to file.")
            return
    
    def run_xfoil(self, airfoil_data):

        alphas = np.linspace(-20, 20, airfoil_data.shape[0])
        cls, cds = foil_data(airfoil_data, alphas, 1e6)
        return np.column_stack((airfoil_data, alphas, cls, cds))

    
    def load_oper_from_file(self, filename):
        try:
            with open(filename, 'r') as f:
                av = json.load(f)
        except FileNotFoundError:
            return False
        self.oper = av['oper']
        self.oper_table.set_values(self.oper)
        self.on_new_oper()
        return True

    def save_oper_to_file(self, filename):
        av = {
            'oper': self.oper
        }
        try:
            with open(filename, 'w') as f:
                json.dump(av, f, indent=4)
        except FileNotFoundError:
            print("Error saving to directory")
            return

    def save_to_fortran(self, fname, avs):

        prop = avs.prop
        oper = avs.oper

        with open(fname, 'w') as f:
            f.write(f"{prop['B']}\n")
            f.write(f"{prop['nr']}\n")
            f.write(f"{prop['nx']}\n")
            f.write(f"{prop['rt']}\n")
            f.write(f"{prop['rh']}\n")

            f.write(f"{oper['V']}\n")
            f.write(f"{oper['Omega']}\n")
            f.write(f"{oper['rho']}\n")

            f.write(f"{prop['foil_path']}\n")
            f.write(f"{prop['rdist']}\n")

            f.write(f"r0_r0\n")
            for r in prop['r0_rt']:
                f.write(f"{r}\n")
            f.write(f"xc\n")
            for c in prop['xc']:
                f.write(f"{c}\n")
            f.write(f"dz\n")
            for dz in prop['dz']:
                f.write(f"{dz}\n")
            f.write(f"chord\n")
            for c in prop['c']:
                f.write(f"{c}\n")
            f.write(f"twist\n")
            for t in prop['twist']:
                f.write(f"{t}\n")
            f.write(f"sweep\n")
            for s in prop['sweep']:
                f.write(f"{s}\n")
                
