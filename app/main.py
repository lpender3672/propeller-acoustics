from PyQt6.QtWidgets import QApplication, QWidget, QGridLayout
from PyQt6.QtCore import QObject, pyqtSignal

from vis import STLViewerWidget
from input import InputWidget
from dists import DistributionsWidget
from results import NoiseResultsWidget, AerodynamicResultsWidget, ResultsTable
from geometry import generate_blade_mesh

import numpy as np
from scipy.interpolate import interp2d
import json
import os
import subprocess
from pathlib import Path

class AppVars(QObject):
    #new_oper = pyqtSignal()
    #new_prop = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.oper = {

        }

        self.prop = {

        }

        self.dist = {

        }

        self.res = {
            
        }

        self.airfoil_data = np.array(
            []
        )

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.av = AppVars()

        self.app_dir = Path(os.path.dirname(os.path.realpath(__file__)))
        
        self.assemble_widgets()
        self.attach_signals()

        av_file = self.app_dir / "app_vars.json"
        self.ft_input_file = self.app_dir / "ft_input.dat"
        self.ft_output_file = self.app_dir / "ft_output.dat"

        if not self.input_widget.load_oper_from_file(av_file):
            pass
            
        self.stl_viewer.load_stl_file("practical/designs/clarkY.stl")#

    def assemble_widgets(self):
        layout = QGridLayout(self)

        self.input_widget = InputWidget(self)
        self.dists_widget = DistributionsWidget(self)
        self.stl_viewer = STLViewerWidget(self)
        self.noise_results_widget = NoiseResultsWidget(self)
        self.aerodynamic_results_widget = AerodynamicResultsWidget(self)
        self.results_table = ResultsTable(self)

        self.dists_widget.setMaximumWidth(500)
        self.dists_widget.setMinimumWidth(400)
        self.input_widget.setMinimumWidth(400)
        self.input_widget.setMinimumHeight(700)
        self.stl_viewer.setMinimumHeight(200)
        self.stl_viewer.setMinimumWidth(400)
        self.noise_results_widget.setMinimumWidth(400)
        self.aerodynamic_results_widget.setMinimumHeight(400)

        
        layout.addWidget(self.input_widget, 0, 0, 2, 1)
        layout.addWidget(self.results_table, 2, 0, 1, 1)
        layout.addWidget(self.dists_widget, 0, 1, 3, 1)
        layout.addWidget(self.stl_viewer, 0, 2, 2, 1)
        layout.addWidget(self.noise_results_widget, 0, 4, 3, 1)
        layout.addWidget(self.aerodynamic_results_widget, 2, 2, 1, 1)

    def attach_signals(self):

        self.input_widget.new_oper.connect(self.update_oper)
        self.input_widget.new_prop.connect(self.update_prop)

        self.dists_widget.new_dist.connect(self.update_prop)
        self.input_widget.save_prop_btn.clicked.connect(self.save_prop)

        self.input_widget.new_prop_from_file.connect(self.on_new_prop_from_file)
    
    def update_prop(self, update_results = True):
        print("Updating prop")
        
        self.av.prop.update(self.input_widget.prop)
        self.av.airfoil_data = self.input_widget.airfoil_data

        self.dists_widget.update_avs(self.av)

        if not self.input_widget.prop_defined:
            return
        # update mesh

        self.stl_viewer.set_mesh(
            generate_blade_mesh(self.av), self.av.prop["B"]
        )

        if not self.input_widget.oper_defined:
            return

        if update_results:
            self.input_widget.save_to_fortran(self.ft_input_file, self.av)
            #subprocess.run(["app/build/propeller_lifting_line.exe", self.ft_input_file])
            self.aerodynamic_results_widget.update_results(self.av)
            self.noise_results_widget.update_results(self.av)
            self.results_table.update_results(self.av)
    
    def on_new_prop_from_file(self):
        self.av.prop.update(self.input_widget.prop)
        self.av.dist.update(self.input_widget.dist)
        self.av.airfoil_data = self.input_widget.airfoil_data

        self.dists_widget.set_dists(self.av) # this will then update the prop

    def update_oper(self, update_results = True):
        print("Updating oper")
        self.av.oper.update(self.input_widget.oper)

        # dont need to update mesh, conditions were not changed
        #self.stl_viewer.set_mesh(
        #    generate_blade_mesh(self.av), self.av.prop["B"]
        #)
        # do need to update results if prop is defined
        if not self.input_widget.prop_defined:
            return # no prop defined
        
        if update_results:
            self.input_widget.save_to_fortran(self.ft_input_file, self.av)
            #subprocess.run(["app/build/propeller_lifting_line.exe", self.ft_input_file])
            self.aerodynamic_results_widget.update_results(self.av)
            self.noise_results_widget.update_results(self.av)
            self.results_table.update_results(self.av)

    def save_prop(self):
        self.input_widget.save_prop_to_file(self.av)
    
    def closeEvent(self, event):
        av_file = os.path.join(self.app_dir, "app_vars.json")
        self.input_widget.save_oper_to_file(av_file)
        event.accept()


def main():
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
