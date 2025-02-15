# Handles Odrive, Serial, and NI-DAQ communication


from PyQt6.QtWidgets import QApplication, QWidget, QGridLayout, QComboBox, QPushButton, QLineEdit, QDialog, QVBoxLayout, QDialogButtonBox, QLabel, QHBoxLayout
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QDoubleValidator

import serial.tools.list_ports

import numpy as np
from pathlib import Path

import pyqtgraph as pg

from collection_threads import (
    SerialReaderThread,
    DAQThread,
    ControllerThread,
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
        self.float_validator = QDoubleValidator(-1e9, 1e9, 10, self)
        self.input_field.setValidator(self.float_validator)
        self.layout.addWidget(self.input_field)
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.layout.addWidget(self.button_box)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        self.setLayout(self.layout)

    def get_value(self):
        text = self.input_field.text()
        return float(text) if text.strip() != "" else None


class ControlWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.speed_box = QLineEdit()
        self.speed_box.setPlaceholderText("Enter speed setpoint")
        self.speed_box.setValidator(QDoubleValidator(-1e9, 1e9, 2, self))

        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)  # Initially disabled

        self.output_box = QLineEdit()
        self.output_box.setReadOnly(True)

        layout = QVBoxLayout()

        layout.addWidget(QLabel("Speed Setpoint:"))
        layout.addWidget(self.speed_box)

        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.start_button)
        buttons_layout.addWidget(self.stop_button)
        layout.addLayout(buttons_layout)

        layout.addWidget(QLabel("Current Speed:"))
        layout.addWidget(self.output_box)

        self.speed_plot = pg.PlotWidget()
        self.speed_curve = self.speed_plot.plot(pen=pg.mkPen(color="g", width=2))

        self.current_plot = pg.PlotWidget()
        self.current_curve = self.current_plot.plot(pen=pg.mkPen(color="g", width=2))
        
        self.motor_data = np.zeros((0,2))

        layout.addWidget(self.speed_plot)

        self.setLayout(layout)

        self.controller = ControllerThread(self)

        self.controller.newSample.connect(self.update_graphs)

        self.start_button.clicked.connect(self.start_control)
        self.stop_button.clicked.connect(self.stop_control)
        self.speed_box.editingFinished.connect(self.set_speed)

        self.controller.start()

    def start_control(self):
        setpoint = self.speed_box.text().strip()

        self.speed_box.setEnabled(False)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        self.controller.startMotor.emit()

    def update_graphs(self, data):

        self.motor_data = np.append(self.motor_data, data, axis=0)

        self.speed_curve.setData(self.motor_data[:,0])
        self.current_curve.setData(self.motor_data[:,1])


    def stop_control(self):
        self.speed_box.setEnabled(True)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

        self.controller.stopMotor.emit()

    def set_speed(self):
        self.controller.setSpeed.emit(
            float(self.speed_box.text()) / 60
        )

    def about_to_quit(self):

        self.controller.stop_thread()
    

class ForceWidget(QWidget):
    def __init__(self, parent = None):
        super().__init__(parent)

        self.thrust_cal_plot = pg.PlotWidget()
        self.torque_cal_plot = pg.PlotWidget()

        self.thrust_cal_curve = self.thrust_cal_plot.plot(pen=pg.mkPen(color="r", width=2))
        self.torque_cal_curve = self.torque_cal_plot.plot(pen=pg.mkPen(color="b", width=2))

        self.thrust_plot = pg.PlotWidget()
        self.torque_plot = pg.PlotWidget()

        self.thrust_plot_curve = self.thrust_plot.plot(pen=pg.mkPen(color="r", width=2))
        self.torque_plot_curve = self.torque_plot.plot(pen=pg.mkPen(color="b", width=2))

        self.serial_thread = None
        self.cal_thread = None

        self.com_selector = QComboBox()
        self.selected_com = None

        self.add_calibration_point_button = QPushButton("Add Calibration Point")

        self.calibration_data = np.zeros((0, 2, 2))
        self.time_buffer = []
        self.amp0_buffer = []
        self.amp1_buffer = []
        self.max_buffer_size = 160

        layout = QGridLayout()
        layout.addWidget(self.com_selector, 0, 0)
        layout.addWidget(self.add_calibration_point_button, 1, 0)
        layout.addWidget(self.thrust_plot, 2, 0)
        layout.addWidget(self.torque_plot, 2, 1)
        layout.addWidget(self.thrust_cal_plot, 3, 0)
        layout.addWidget(self.torque_cal_plot, 3, 1)

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
        try:
            self.serial_thread = SerialReaderThread(self.selected_com, 115200)
            self.serial_thread.data_received.connect(self.update_live_plots) # Maybe dont update every new datapoint
            self.serial_thread.start()
            print("thread started")
        except:
            print("failed to start thread")
            pass

    def update_live_plots(self, time_val, amp0_val, amp1_val):
        self.time_buffer.append(time_val)
        self.amp0_buffer.append(amp0_val)
        self.amp1_buffer.append(amp1_val)

        if len(self.time_buffer) > self.max_buffer_size:
            self.time_buffer = self.time_buffer[-self.max_buffer_size:]
            self.amp0_buffer = self.amp0_buffer[-self.max_buffer_size:]
            self.amp1_buffer = self.amp1_buffer[-self.max_buffer_size:]
        
        if len(self.time_buffer) == len(self.amp0_buffer):
            self.thrust_plot_curve.setData(self.time_buffer, self.amp0_buffer)

        if len(self.time_buffer) == len(self.amp1_buffer):
            self.torque_plot_curve.setData(self.time_buffer, self.amp1_buffer)

    def start_cal(self):
        # get mass of calibration weight from user
        if not self.selected_com:
            return

        if self.serial_thread:
            self.serial_thread.stop()

        self.cal_thread = SerialThreadWrapper(self.selected_com, 115200, 40)
        self.cal_thread.data_received.connect(self.cal_plot_update)

    def cal_plot_update(self, sample):

        if not self.serial_thread.running:
            self.serial_thread.running = True
            self.serial_thread.start()

        _, raw_thrust, raw_torque = np.mean(sample, axis=0)

        mcal_dialog = FloatInputDialog("Enter Calibration Mass", "Enter the mass of the calibration weight in kg:")
        mcal_dialog.exec()

        mcal = mcal_dialog.get_value()
        if mcal is None:
            return
        
        mes_thrust = mcal * gravity * np.sqrt(2)/2
        mes_torque = mcal * offset * gravity * np.sqrt(2)/2

        new_data = np.array([[[raw_thrust, raw_torque],
                              [mes_thrust, mes_torque]]])
        self.calibration_data = np.concatenate((self.calibration_data, new_data))

        # add points to calibration plots
        self.thrust_cal_curve.setData(self.calibration_data[:,:,0], pen='r')
        self.torque_cal_curve.setData(self.calibration_data[:,:,1], pen='b')

        # TODO: save calibration data
        print(self.calibration_data)
    
    def interpolate_calibration(self, raw_data):
        # curve fit to calibration data
        thrustcoeffs = np.polyfit(self.calibration_data[:,0,0], self.calibration_data[:,0,1], 1)
        torquecoeffs = np.polyfit(self.calibration_data[:,1,0], self.calibration_data[:,1,1], 1)

        thrust = np.polyval(thrustcoeffs, raw_data[0])
        torque = np.polyval(torquecoeffs, raw_data[1])

        return np.array([thrust, torque])

        
    def about_to_quit(self):
        if self.serial_thread:
            self.serial_thread.stop()
        
        # maybe save calibration data

class AudioWidget(QWidget):
    def __init__(self, parent = None):
        super().__init__(parent)

        self.signal_plot = pg.PlotWidget()
        self.spectrum_plot = pg.PlotWidget()

        self.signal_curve = self.signal_plot.plot(pen=pg.mkPen(color="r", width=2))
        self.spectrum_curve = self.spectrum_plot.plot(pen=pg.mkPen(color="r", width=2))

        self.daq_thread = None

        self.bpf = 167
        self.max_harmonic = 10
        self.nyquist_factor = 2

        self.sample_freq = self.nyquist_factor * self.max_harmonic * self.bpf
        self.buffer_freq = self.nyquist_factor * self.bpf

        layout = QGridLayout()

        layout.addWidget(self.signal_plot, 0, 0)
        layout.addWidget(self.spectrum_plot, 1, 0)

        self.setLayout(layout)

        self.start_daq()

        
        
    def start_daq(self):
        
        self.daq_thread = DAQThread(
            self, 
            self.sample_freq,
            self.buffer_freq
            )
        self.daq_thread.newSample.connect(self.update_signal_plot)
        self.daq_thread.errorOccurred.connect(self.error_occurred)
        self.data_buffer = np.zeros(self.buffer_freq)

        self.daq_thread.add_channel()
        self.daq_thread.start()

    def update_signal_plot(self, data):
        if data.shape[0] > 0:
            channel_data = data[0]
            self.data_buffer = np.roll(self.data_buffer, -len(channel_data))
            self.data_buffer[-len(channel_data):] = channel_data
            self.signal_curve.setData(self.data_buffer)
            self.update_spectrum_plot(self.data_buffer)
    
    def error_occurred(self, error):
        print(error)

    def update_spectrum_plot(self, sample):

        dtft = np.fft.fft(sample)
        dt = 1 / (self.nyquist_factor * self.max_harmonic * self.bpf)
        freqs = np.fft.fftfreq(len(sample), d=dt)

        self.spectrum_curve.setData(freqs, np.abs(dtft))

    def about_to_quit(self):
        self.daq_thread.stop()

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.control_widget = ControlWidget()
        self.force_widget = ForceWidget()
        self.audio_widget = AudioWidget()

        QApplication.instance().aboutToQuit.connect(self.about_to_quit)

        layout = QGridLayout()
        layout.addWidget(self.control_widget, 0, 0)
        layout.addWidget(self.force_widget, 0, 1)
        layout.addWidget(self.audio_widget, 0, 2)

        self.setLayout(layout)

    def resizeEvent(self, event):
        width = self.width() // 3
        self.control_widget.setFixedWidth(width)
        self.force_widget.setFixedWidth(width)
        super().resizeEvent(event)

    def about_to_quit(self):

        self.control_widget.about_to_quit()
        self.force_widget.about_to_quit()
        self.audio_widget.about_to_quit()

    

if __name__ == "__main__":
    app = QApplication([])

    window = MainWindow()
    window.show()

    app.exec()
