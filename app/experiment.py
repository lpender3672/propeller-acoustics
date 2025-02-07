# Handles Odrive, Serial, and NI-DAQ communication


from PyQt6.QtWidgets import QApplication, QWidget, QGridLayout, QComboBox, QPushButton, QLineEdit, QDialog, QVBoxLayout, QDialogButtonBox, QLabel
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QDoubleValidator

import serial.tools.list_ports

import numpy as np
from pathlib import Path

import pyqtgraph as pg

from collection_threads import (
    SerialReaderThread,
    DAQThread,
    Controller,
    SerialThreadWrapper
)

gravity = 9.81 # m/s^2
offset = 0.1 # m

class FloatInputDialog(QDialog):
    def __init__(self, title="Float Input", prompt="Enter a float value:", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.layout = QVBoxLayout()

        self.label = QLabel(prompt)
        self.layout.addWidget(self.label)
        self.input_field = QLineEdit()
        self.float_validator = QDoubleValidator()
        self.input_field.setValidator(self.float_validator)
        self.layout.addWidget(self.input_field)
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.layout.addWidget(self.button_box)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        self.setLayout(self.layout)

    def get_value(self):
        return float(self.input_field.text()) if self.input_field.text() else None


class ControlWidget(QWidget):
    def __init__(self, parent = None):
        super().__init__(parent)

class ForceWidget(QWidget):
    def __init__(self, parent = None):
        super().__init__(parent)

        self.thrust_cal_curve = pg.PlotWidget()
        self.torque_cal_curve = pg.PlotWidget()

        self.serial_thread = None
        self.cal_thread = None

        self.com_selector = QComboBox()
        self.selected_com = None

        self.add_calibration_point_button = QPushButton("Add Calibration Point")

        self.calibration_data = np.zeros((0, 2, 2))

        layout = QGridLayout()
        layout.addWidget(self.com_selector, 0, 0)
        layout.addWidget(self.add_calibration_point_button, 1, 0)
        layout.addWidget(self.thrust_cal_curve, 2, 0)
        layout.addWidget(self.torque_cal_curve, 2, 0)

        self.setLayout(layout)

        self.com_selector.currentIndexChanged.connect(self.com_selected)
        self.add_calibration_point_button.clicked.connect(self.start_cal)

        self.scan_com_ports()


    def scan_com_ports(self):
        ports = serial.tools.list_ports.comports()
        self.com_selector.currentIndexChanged.disconnect()
        self.com_selector.clear()
        for port in ports:
            self.com_selector.addItem(f"{port.device} ({port.description})")
        if self.com_selector.count() == 0:
            self.com_selector.addItem("No COM ports found")
            self.selected_com = None
        else:
            self.com_selector.currentIndexChanged.connect(self.com_selected)
            self.com_selected(0)

    def com_selected(self, idx):
        self.selected_com = self.com_selector.currentText().split(" ")[0]     
        # maybe check data on this port to see if it's valid

    def start_cal(self):
        # get mass of calibration weight from user
        if not self.selected_com:
            return

        if self.serial_thread:
            self.serial_thread.stop()

        self.cal_thread = SerialThreadWrapper(self.selected_com, 115200, 40)
        self.cal_thread.data_received.connect(self.cal_plot_update)

    def cal_plot_update(self, sample):

        _, raw_thrust, raw_torque = np.mean(sample, axis=0)

        mcal_dialog = FloatInputDialog("Enter Calibration Mass", "Enter the mass of the calibration weight in kg:")
        mcal_dialog.exec()

        mcal = mcal_dialog.get_value()
        if not mcal:
            return
        
        mes_thrust = mcal * gravity * np.sqrt(2)/2
        mes_torque = mcal * offset * gravity * np.sqrt(2)/2

        new_data = np.array([[[raw_thrust, raw_torque],
                              [mes_thrust, mes_torque]]])
        self.calibration_data = np.concatenate((self.calibration_data, new_data))

        # add points to calibration plots
        self.thrust_cal_curve.plot(self.calibration_data[:,0,0], self.calibration_data[:,0,1], pen='r')
        self.torque_cal_curve.plot(self.calibration_data[:,1,0], self.calibration_data[:,1,1], pen='b')

        # TODO: save calibration data

    def about_to_close(self):
        if self.serial_thread:
            self.serial_thread.stop()
        
        # maybe save calibration data

class AudioWidget(QWidget):
    def __init__(self, parent = None):
        super().__init__(parent)

        self.signal_plot = pg.PlotWidget()
        self.spectrum_plot = pg.PlotWidget()

        self.daq_thread = None

        self.bpf = 167
        self.max_harmonic = 10
        self.nyquist_factor = 2

        layout = QGridLayout()

        layout.addWidget(self.signal_plot, 0, 0)
        layout.addWidget(self.spectrum_plot, 1, 0)

        self.setLayout(layout)
        
    def start_daq(self):
        self.daq_thread = DAQThread(
            self, 
            self.nyquist_factor * self.max_harmonic * self.bpf,
            self.nyquist_factor * self.bpf
            )
        self.daq_thread.newSample.connect(self.update_plot)
        self.daq_thread.errorOccurred.connect(self.error_occurred)
        self.daq_thread.start()
    
    def error_occurred(self, error):
        print(error)

    def update_plot(self, sample):
        self.signal_plot.plot(sample)

        dtft = np.fft.fft(sample)
        dt = 1 / (self.nyquist_factor * self.max_harmonic * self.bpf)
        freqs = np.fft.fftfreq(len(sample), d=dt)

        self.spectrum_plot.plot(freqs, np.abs(dtft))

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.control_widget = ControlWidget()
        self.force_widget = ForceWidget()
        self.audio_widget = AudioWidget()

        layout = QGridLayout()
        layout.addWidget(self.control_widget, 0, 0)
        layout.addWidget(self.force_widget, 1, 0)
        layout.addWidget(self.audio_widget, 1, 1, 2, 1)

        self.setLayout(layout)

    

if __name__ == "__main__":
    app = QApplication([])

    window = MainWindow()
    window.show()

    app.exec()
