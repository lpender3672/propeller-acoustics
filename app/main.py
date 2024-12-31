from PyQt6.QtWidgets import QApplication, QWidget, QGridLayout
from PyQt6.QtCore import QObject, pyqtSignal

from vis import STLViewerWidget
from input import InputWidget
from dists import DistributionsWidget
from results import NoiseResultsWidget
from geometry import generate_blade_mesh

import numpy as np
from scipy.interpolate import interp2d
import json
import os

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

        self.airfoil_data = np.array(
            []
        )

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.av = AppVars()

        self.app_dir = os.path.dirname(os.path.realpath(__file__))
        
        self.assemble_widgets()
        self.attach_signals()

        av_file = os.path.join(self.app_dir, "app_vars.json")
        if self.input_widget.load_oper_from_file(av_file):
            pass
            
        self.stl_viewer.load_stl_file("practical/designs/clarkY.stl")#

    def assemble_widgets(self):
        layout = QGridLayout(self)

        self.input_widget = InputWidget(self)
        self.dists_widget = DistributionsWidget(self)
        self.stl_viewer = STLViewerWidget(self)

        self.aero_results_widget = NoiseResultsWidget(self)
        
        layout.addWidget(self.input_widget, 0, 0, 3, 1)
        layout.addWidget(self.dists_widget, 0, 1, 3, 1)
        layout.addWidget(self.stl_viewer, 0, 2, 2, 2)
        layout.addWidget(self.aero_results_widget, 2, 2, 1, 2)

    def attach_signals(self):

        self.input_widget.new_oper.connect(self.update_oper)
        self.input_widget.new_prop.connect(self.update_prop)

        self.dists_widget.new_dist.connect(self.update_prop)
        self.input_widget.save_prop_btn.clicked.connect(self.save_prop)

        self.input_widget.new_prop_from_file.connect(self.on_new_prop_from_file)
    
    def update_prop(self):
        print("Updating prop")
        
        self.av.prop.update(self.input_widget.prop)
        self.av.airfoil_data = self.input_widget.airfoil_data

        self.dists_widget.update_avs(self.av)

        self.stl_viewer.set_mesh(
            generate_blade_mesh(self.av), self.av.prop["B"]
        )
    
    def on_new_prop_from_file(self):
        self.av.prop.update(self.input_widget.prop)
        self.av.dist.update(self.input_widget.dist)
        self.av.airfoil_data = self.input_widget.airfoil_data

        self.dists_widget.set_dists(self.av)

    def update_oper(self):
        print("Updating oper")
        self.av.oper.update(self.input_widget.oper)

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
