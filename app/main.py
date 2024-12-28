from PyQt6.QtWidgets import QApplication, QWidget, QGridLayout
from PyQt6.QtCore import QObject, pyqtSignal

from vis import STLViewerWidget
from input import InputWidget
from dists import DistributionPlotWidget

import numpy as np
from scipy.interpolate import interp2d
import json
import os

class AppVars(QObject):
    new_oper = pyqtSignal()
    new_prop = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.oper = {

        }

        self.prop = {

        }

    def load_oper_from_file(self, filename):
        try:
            with open(filename, 'r') as f:
                av = json.load(f)
        except FileNotFoundError:
            return False
        self.oper = av['oper']
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

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.av = AppVars()

        self.app_dir = os.path.dirname(os.path.realpath(__file__))
        
        self.assemble_widgets()
        self.attach_signals()

        av_file = os.path.join(self.app_dir, "app_vars.json")
        if self.av.load_oper_from_file(av_file):
            self.input_widget.oper_table.set_values(self.av.oper)
            
        self.stl_viewer.load_stl_file("practical/designs/clarkY.stl")#

    def assemble_widgets(self):
        layout = QGridLayout(self)

        self.input_widget = InputWidget(self)
        self.stl_viewer = STLViewerWidget()
        
        self.thickness_plot = DistributionPlotWidget()
        self.chord_plot = DistributionPlotWidget()
        self.sweep_plot = DistributionPlotWidget()

        layout.addWidget(self.input_widget, 0, 0, 3, 1)

        layout.addWidget(self.thickness_plot, 0, 1, 1, 1)
        layout.addWidget(self.chord_plot, 1, 1, 1, 1)
        layout.addWidget(self.sweep_plot, 2, 1, 1, 1)

        layout.addWidget(self.stl_viewer, 0, 2, 3, 1)

    def attach_signals(self):

        self.input_widget.new_oper.connect(self.update_oper)
        self.input_widget.new_prop.connect(self.update_prop)

        self.thickness_plot.new_dist.connect(self.update_prop)
        self.chord_plot.new_dist.connect(self.update_prop)
        self.sweep_plot.new_dist.connect(self.update_prop)
    
    def update_prop(self):
        print("Updating prop")
        
        self.av.prop = self.input_widget.prop_table.parse_values()
        self.av.prop['HX'] = self.thickness_plot.get_distribution(self.av.prop['r0_rt'])
        self.av.prop['c'] = self.chord_plot.get_distribution(self.av.prop['r0_rt'])
        self.av.prop['sweep'] = self.sweep_plot.get_distribution(self.av.prop['r0_rt'])

        # fLX and fDX


    def update_oper(self):
        print("Updating oper")
        
        self.av.oper = self.input_widget.oper_table.parse_values()

    
    def closeEvent(self, event):
        av_file = os.path.join(self.app_dir, "app_vars.json")
        self.av.save_oper_to_file(av_file)
        event.accept()


def main():
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
