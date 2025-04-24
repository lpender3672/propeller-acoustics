# Handles Odrive, Serial, and NI-DAQ communication


import os
import time
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import serial.tools.list_ports
from collection_threads import ControllerThread, DAQThread, SerialReaderThread
from PyQt6.QtCore import QObject, QTimer, pyqtSignal
from PyQt6.QtGui import QDoubleValidator
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from routines import append_audiof_to_metaf, load_prop_from_file

gravity = 9.81  # m/s^2
offset = 0.03  # 3 cm


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
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
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

        self.calibrate_button = QPushButton("Calibrate")

        self.output_box = QLineEdit()
        self.output_box.setReadOnly(True)

        layout = QVBoxLayout()

        layout.addWidget(self.calibrate_button)

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
        self.temperature_curve = self.current_plot.plot(
            pen=pg.mkPen(color="b", width=2)
        )

        self.sample_rate = 200
        self.sample_buffer_size = 10
        self.graph_buffer_size = 10 * self.sample_rate  # 10 seconds
        self.motor_data = np.zeros((self.graph_buffer_size, 4))
        self.motor_data[:, 1:] = np.nan

        layout.addWidget(self.speed_plot)
        layout.addWidget(self.current_plot)

        self.setLayout(layout)

        self.controller = ControllerThread(
            self, self.sample_rate, self.sample_buffer_size
        )

        self.controller.newSample.connect(self.update_graphs)
        self.controller.errorOccurred.connect(self.error_occurred)

        self.start_button.clicked.connect(self.start_control)
        self.stop_button.clicked.connect(self.stop_control)
        self.speed_box.editingFinished.connect(self.set_speed)

        self.calibrate_button.clicked.connect(self.controller.calibrateMotor.emit)

        self.controller.start()

    def error_occurred(self, error):
        print(error)

    def start_control(self):

        if not self.controller.thread_running:
            return

        self.speed_box.setEnabled(False)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        self.controller.startMotor.emit()

    def update_graphs(self, new_data):

        self.motor_data = np.roll(self.motor_data, -new_data.shape[0], axis=0)
        self.motor_data[-new_data.shape[0]:] = new_data

        self.speed_curve.setData(self.motor_data[:, 0], self.motor_data[:, 1])
        self.current_curve.setData(self.motor_data[:, 0], self.motor_data[:, 2])
        self.temperature_curve.setData(self.motor_data[:, 0], self.motor_data[:, 3])

    def stop_control(self):
        self.speed_box.setEnabled(True)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

        self.controller.stopMotor.emit()

    def set_speed(self):
        self.controller.setSpeed.emit(float(self.speed_box.text()) / 60)

    def about_to_quit(self):

        self.controller.stop_thread()


class ForceWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.app_dir = Path(os.path.dirname(os.path.realpath(__file__)))
        self.cal_file = self.app_dir / "calibration.npy"

        self.thrust_cal_plot = pg.PlotWidget()
        self.torque_cal_plot = pg.PlotWidget()

        self.thrust_cal_points = self.thrust_cal_plot.plot(
            pen=pg.mkPen(color="r", width=2)
        )
        self.thrust_cal_fit = self.thrust_cal_plot.plot(
            pen=pg.mkPen(color="r", width=2)
        )
        self.thrust_cal_unc = self.thrust_cal_plot.plot(
            pen=pg.mkPen(color="r", width=2)
        )
        self.torque_cal_points = self.torque_cal_plot.plot(
            pen=pg.mkPen(color="b", width=2)
        )
        self.torque_cal_fit = self.torque_cal_plot.plot(
            pen=pg.mkPen(color="b", width=2)
        )
        self.torque_cal_unc = self.torque_cal_plot.plot(
            pen=pg.mkPen(color="b", width=2)
        )

        self.thrust_plot = pg.PlotWidget()
        self.torque_plot = pg.PlotWidget()

        self.thrust_plot_curve = self.thrust_plot.plot(pen=pg.mkPen(color="r", width=2))
        self.torque_plot_curve = self.torque_plot.plot(pen=pg.mkPen(color="b", width=2))

        self.serial_thread = None

        self.com_selector = QComboBox()
        self.selected_com = None

        self.clear_calibration_button = QPushButton("Clear Calibration Data")

        self.add_calibration_point_button = QPushButton("Add Calibration Point")

        self.sample_rate = 10  # Load cell amp is stuck at 10Hz
        self.max_buffer_size = 160
        self.force_data = np.zeros((self.max_buffer_size, 3))
        self.force_data[:, 1:] = np.nan
        self.t0 = None

        self.layout = QGridLayout()
        self.layout.addWidget(self.com_selector, 0, 0)
        self.layout.addWidget(self.add_calibration_point_button, 1, 0)
        self.layout.addWidget(self.clear_calibration_button, 1, 1)
        self.layout.addWidget(self.thrust_plot, 2, 0)
        self.layout.addWidget(self.torque_plot, 2, 1)
        self.layout.addWidget(self.thrust_cal_plot, 3, 0)
        self.layout.addWidget(self.torque_cal_plot, 3, 1)

        self.setLayout(self.layout)

        self.com_selector.currentIndexChanged.connect(self.com_selected)
        self.add_calibration_point_button.clicked.connect(self.start_cal)
        self.clear_calibration_button.clicked.connect(self.clear_calibration)

        try:
            self.calibration_data = np.load(self.cal_file)
        except FileNotFoundError:
            self.calibration_data = np.zeros((0, 2, 2))
        else:
            self.update_cal_graph()

        self.scan_com_ports()

    def clear_calibration(self):
        self.calibration_data = np.zeros((0, 2, 2))
        np.save(self.cal_file, self.calibration_data)
        self.update_cal_graph()

    def scan_com_ports(self):
        print("Scanning COM ports")
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
            i = 0
            while (
                i < len(ports)
                and self.serial_thread is not None
                and self.serial_thread.running
            ):
                self.com_selector.setCurrentIndex(i)
                i += 1

    def com_selected(self, idx):
        self.selected_com = self.com_selector.currentText().split(" ")[0]
        # maybe check data on this port to see if it's valid
        try:
            self.serial_thread = SerialReaderThread(self.selected_com, 115200)
            assert self.serial_thread is not None
            assert self.serial_thread.running
        except Exception as e:
            self.serial_thread = None
            print("failed to start thread:,", e)
            return

        self.serial_thread.data_received.connect(
            self.update_live_plots
        )  # Maybe dont update every new datapoint
        self.serial_thread.error_occurred.connect(self.error_occurred)
        self.serial_thread.start()

    def error_occurred(self, error):
        if self.serial_thread is None:
            return

        self.serial_thread.stop()
        self.serial_thread.wait()
        self.serial_thread = None
        print(error)

    def update_live_plots(self, new_data):
        time_val, amp0_val, amp1_val = new_data

        if self.t0 is None:
            self.t0 = time_val

        newpoint = [time_val - self.t0, amp0_val, amp1_val]
        self.force_data = np.roll(self.force_data, -1, axis=0)
        self.force_data[-1] = newpoint

        self.thrust_plot_curve.setData(self.force_data[:, 0], self.force_data[:, 1])
        self.torque_plot_curve.setData(self.force_data[:, 0], self.force_data[:, 2])

    def start_cal(self):
        # get mass of calibration weight from user
        if self.serial_thread is None:
            return
        self.serial_thread.finishedLogging.connect(self.cal_plot_update)
        self.serial_thread.startLogging.emit(40)  # 40 samples

    def cal_plot_update(self, sample):
        self.serial_thread.finishedLogging.disconnect(self.cal_plot_update)
        self.serial_thread
        _, raw_thrust, raw_torque = np.mean(sample, axis=0)

        mcal_dialog = FloatInputDialog(
            "Enter Calibration Mass", "Enter the mass of the calibration weight in kg:"
        )
        mcal_dialog.exec()

        mcal = mcal_dialog.get_value()
        if mcal is None:
            return

        mes_thrust = mcal * gravity  # np.sqrt(2)/2
        mes_torque = mcal * offset * gravity * 0  # * np.sqrt(2)/2

        new_data = np.array([[[raw_thrust, raw_torque], [mes_thrust, mes_torque]]])
        self.calibration_data = np.concatenate((self.calibration_data, new_data))
        np.save(self.cal_file, self.calibration_data)
        self.update_cal_graph()

    def update_cal_graph(self):
        self.thrust_cal_points.setData(
            self.calibration_data[:, :, 0], pen=None, symbol="o", symbolBrush="r"
        )
        self.torque_cal_points.setData(
            self.calibration_data[:, :, 1], pen=None, symbol="o", symbolBrush="b"
        )

        if self.calibration_data.shape[0] < 2:
            return

        thrust_x = np.linspace(
            np.min(self.calibration_data[:, 0, 0]),
            np.max(self.calibration_data[:, 0, 0]),
            50,
        )
        torque_x = np.linspace(
            np.min(self.calibration_data[:, 0, 1]),
            np.max(self.calibration_data[:, 0, 1]),
            50,
        )
        thrust_y, torque_y = self.interpolate_calibration([thrust_x, torque_x])

        self.thrust_cal_fit.setData(thrust_x, thrust_y)
        self.torque_cal_fit.setData(torque_x, torque_y)

    def interpolate_calibration(self, raw_data):
        # curve fit to calibration data
        thrustcoeffs = np.polyfit(
            self.calibration_data[:, 0, 0], self.calibration_data[:, 1, 0], 1
        )
        torquecoeffs = np.polyfit(
            self.calibration_data[:, 0, 1], self.calibration_data[:, 1, 1], 1
        )

        thrust = np.polyval(thrustcoeffs, raw_data[0])
        torque = np.polyval(torquecoeffs, raw_data[1])

        return np.array([thrust, torque])

    def resizeEvent(self, event):
        total_width = self.width()  # Get the full width of the widget
        half_width = total_width // 2  # Split equally

        # Enforce 50/50 width distribution
        self.thrust_plot.setFixedWidth(half_width)
        self.torque_plot.setFixedWidth(half_width)
        self.thrust_cal_plot.setFixedWidth(half_width)
        self.torque_cal_plot.setFixedWidth(half_width)

        super().resizeEvent(event)

    def about_to_quit(self):
        if self.serial_thread:
            self.serial_thread.stop()
        # maybe save calibration data


class AudioWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.signal_plot = pg.PlotWidget()
        self.spectrum_plot = pg.PlotWidget()

        self.harmonic_curve = self.signal_plot.plot(pen=pg.mkPen(color="r", width=2))
        self.spectrum_curve = self.spectrum_plot.plot(pen=pg.mkPen(color="r", width=2))

        self.channel_selector = QComboBox()
        self.channel_selector.currentIndexChanged.connect(self.channel_selected)

        self.daq_thread = None

        self.bpf = 167
        self.max_harmonic = 10
        self.nyquist_factor = 2.2

        self.nchannels = 7

        # self.sample_freq = self.nyquist_factor * self.max_harmonic * self.bpf
        self.sample_freq = 51200.0
        self.buffer_freq = 4000  # self.nyquist_factor * self.bpf

        self.window = np.hanning(self.buffer_freq)

        layout = QGridLayout()

        layout.addWidget(self.channel_selector, 0, 0)
        layout.addWidget(self.spectrum_plot, 1, 0)
        layout.addWidget(self.signal_plot, 2, 0)

        self.setLayout(layout)

        self.start_daq()

    def channel_selected(self, idx):
        pass

    def start_daq(self):

        self.daq_thread = DAQThread(self, self.sample_freq, self.buffer_freq)
        self.daq_thread.newSample.connect(self.update_signal_plot)
        self.daq_thread.errorOccurred.connect(self.error_occurred)
        self.data_buffer = np.zeros(self.buffer_freq)

        for i in range(self.nchannels):
            if not self.daq_thread.add_channel():
                self.daq_thread = None
                print("Failed to add channel")
                return
            else:
                self.channel_selector.addItem(self.daq_thread.channel_names[-1])

        self.daq_thread.start()

    def update_signal_plot(self, data):
        if data.shape[0] > 0:
            idx = self.channel_selector.currentIndex()
            channel_data = data[idx]
            self.data_buffer = np.roll(self.data_buffer, -channel_data.shape[0], axis=0)
            self.data_buffer[-channel_data.shape[0]:] = channel_data

            self.update_spectrum_plot(self.data_buffer)

    def error_occurred(self, error):
        print(error)

    def update_spectrum_plot(self, sample):

        if sample.ndim > 1:
            idx = self.channel_selector.currentIndex()
            sample = sample[:, idx]

        windowed_data = sample * self.window

        zero_padding = 2 ** np.ceil(np.log2(windowed_data.shape[0]) + 1).astype(int)

        dtft = np.fft.rfft(windowed_data, n=zero_padding)
        dt = 1 / self.sample_freq
        freqs = np.fft.rfftfreq(zero_padding, d=dt)

        dBmag = 20 * np.log10(np.maximum(np.abs(dtft), 1e-10))
        self.spectrum_curve.setData(freqs, dBmag)

        # now we need to integrate the harmonics
        prop = self.parent().test_widget.prop
        if prop is None:
            return

        motor_data = self.parent().control_widget.motor_data
        if motor_data is None:
            return

        avg_speed = np.mean(motor_data[-10:, 1]) * -1 / 60
        bpf = avg_speed * prop["B"]

        if bpf < 1:
            return

        harmonics = np.arange(bpf, freqs[-1], bpf)

        intergrated_harmonics = np.zeros((harmonics.shape[0]))

        aband = 10

        for j in range(harmonics.shape[0]):

            mask = (freqs >= harmonics[j] - aband) & (freqs <= harmonics[j] + aband)
            fqharm = freqs[mask]
            ftharm = dtft[mask]

            # trapz power spectral density
            intergrated_harmonics[j] = np.trapz(np.abs(ftharm), fqharm**2, axis=0)

        intergrated_harmonics_dB = 20 * np.log10(
            np.maximum(intergrated_harmonics, 1e-10)
        )

        max_harmonic = 20
        harmonic_num = harmonics // bpf
        ms = harmonic_num[harmonic_num < max_harmonic]
        plot_harmonics = intergrated_harmonics_dB[harmonic_num < max_harmonic]

        self.harmonic_curve.setData(ms, plot_harmonics)

        self.signal_plot.getViewBox().enableAutoRange(axis="y", enable=False)
        self.signal_plot.getViewBox().setYRange(50, 100)

    def about_to_quit(self):
        if self.daq_thread:
            self.daq_thread.stop()


class TestWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.layout = QGridLayout()

        self.prop_label = QLabel("Propeller")
        self.prop_path = QLineEdit()
        self.load_prop_button = QPushButton("...")
        self.results_label = QLabel("Results")
        self.results_directory_path = QLineEdit()
        self.results_directory_path.setReadOnly(True)
        self.microphone_label = QLabel("Microphone Layout")
        self.microphone_layout_path = QLineEdit()
        self.microphone_layout_path.setReadOnly(True)
        self.microphone_layout_button = QPushButton("...")
        self.select_results_directory_button = QPushButton("...")

        self.app_dir = Path(os.path.dirname(os.path.realpath(__file__)))
        self.results_dir = self.app_dir / "results"

        self.aero_pyramid_test_start_button = QPushButton("Start Aero Pyramid")
        self.audio_pyramid_test_start_button = QPushButton("Start Audio Pyramid")

        self.record_ten_seconds_button = QPushButton("Record 1s audio")

        self.layout.addWidget(self.prop_label, 0, 0)
        self.layout.addWidget(self.prop_path, 0, 1)
        self.layout.addWidget(self.load_prop_button, 0, 2)
        self.layout.addWidget(self.results_label, 1, 0)
        self.layout.addWidget(self.results_directory_path, 1, 1)
        self.layout.addWidget(self.select_results_directory_button, 1, 2)
        self.layout.addWidget(self.microphone_label, 2, 0)
        self.layout.addWidget(self.microphone_layout_path, 2, 1)
        self.layout.addWidget(self.microphone_layout_button, 2, 2)
        self.layout.addWidget(self.aero_pyramid_test_start_button, 3, 0, 1, 2)
        self.layout.addWidget(self.audio_pyramid_test_start_button, 4, 0, 1, 2)
        self.layout.addWidget(self.record_ten_seconds_button, 5, 0, 1, 2)

        self.load_prop_button.clicked.connect(self.on_load_prop_clicked)
        self.select_results_directory_button.clicked.connect(
            self.on_select_results_directory
        )
        self.microphone_layout_button.clicked.connect(self.on_select_microhpone_layout)
        self.aero_pyramid_test_start_button.clicked.connect(
            self.on_aero_pyramid_test_started
        )
        self.audio_pyramid_test_start_button.clicked.connect(
            self.on_audio_pyramid_test_started
        )

        self.record_ten_seconds_button.clicked.connect(self.record_ten_seconds)

        self.setLayout(self.layout)

        self.step_timer = QTimer()
        self.step_timer.timeout.connect(self.aero_pyramid_step)

        self.pyramid_steps = 16
        self.min_pyramid_speed = 3000
        self.max_pyramid_speed = 16000
        # self.pyramid_type = 'linear'
        self.pyramid_type = "logarithmic"

        if self.pyramid_type == "logarithmic":
            self.speed_pyramid = np.logspace(
                np.log10(self.min_pyramid_speed),
                np.log10(self.max_pyramid_speed),
                self.pyramid_steps // 2,
            )
            self.speed_pyramid = np.concatenate(
                (self.speed_pyramid[:-1], self.speed_pyramid[::-1])
            )

        else:
            self.speed_pyramid = np.linspace(
                self.min_pyramid_speed, self.max_pyramid_speed, self.pyramid_steps // 2
            )
            self.speed_pyramid = np.concatenate(
                (self.speed_pyramid[:-1], self.speed_pyramid[::-1])
            )

        print(self.speed_pyramid)

        self.idx = 0

        self.prop = None

    def get_files(self):
        formatted_time = time.strftime("%Y-%m-%d-%H-%M-%S")

        if self.prop is None:
            audio_file = self.results_dir / f"audio_{formatted_time}.bin"
            meta_file = self.results_dir / f"meta_{formatted_time}.npy"
            aero_file = self.results_dir / f"aero_{formatted_time}.npz"
        else:
            directory = self.results_dir / self.prop["name"]
            if not directory.exists():
                directory.mkdir()
            audio_file = directory / f"audio_{formatted_time}.bin"
            meta_file = directory / "meta_data.npy"
            aero_file = directory / f"aero_{formatted_time}.npz"

        return audio_file, meta_file, aero_file

    def record_ten_seconds(self):

        if self.parent().audio_widget.daq_thread is None:
            return

        audio_file, _, _ = self.get_files()
        buffer_freq = self.parent().audio_widget.buffer_freq
        sample_freq = self.parent().audio_widget.sample_freq
        duration = 1
        naubufs = np.ceil(duration * sample_freq / buffer_freq).astype(int)
        self.parent().audio_widget.daq_thread.startLogging.emit(
            str(audio_file), naubufs
        )
        self.parent().audio_widget.daq_thread.finishedLogging.connect(
            self.finished_recording
        )

        # start recording speed as well:
        self.nspbufs = 10
        self.idx = 0
        speed_buffer_size = self.parent().control_widget.sample_buffer_size
        self.motor_data = np.zeros((1, self.nspbufs * speed_buffer_size, 4))
        self.parent().control_widget.controller.finishedLogging.connect(
            self.speed_callback
        )
        self.parent().control_widget.controller.startLogging.emit(self.nspbufs)

        self.audio_start_time = time.time()

    def finished_recording(self, fname):
        dt = time.time() - self.audio_start_time
        _, metaf, _ = self.get_files()
        _, speed, current, temp = np.mean(
            self.motor_data[0, :, :], axis=0
        )  # [thetime, vel, current, temperature]

        micpath = str(Path(self.microphone_layout_path.text()))
        append_audiof_to_metaf(fname, metaf, [speed, current, temp, micpath])

        self.parent().audio_widget.daq_thread.finishedLogging.disconnect(
            self.finished_recording
        )

        finished_dialog = QMessageBox()
        finished_dialog.setText(f"Finished recording audio in f{dt}")
        finished_dialog.exec()

    def on_load_prop_clicked(self):
        print("Load Prop button clicked")

        path = QFileDialog.getOpenFileName(
            self,
            "Select propeller file",
            "app/props",
            filter="Propeller files (*.prop)",
        )[0]
        if not path:
            return

        try:
            prop = load_prop_from_file(path)
        except BaseException:
            QMessageBox.critical(
                self, "Error", "Failed to load propeller data from file."
            )
            return

        self.prop_path.setText(path)
        self.prop = prop

        self.prop["name"] = Path(path).name

    def on_select_results_directory(self):
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Results Directory", str(self.results_dir)
        )
        if dir_path:
            self.results_directory_path.setText(dir_path)

    def on_select_microhpone_layout(self):
        layout_path = QFileDialog.getOpenFileName(
            self,
            "Select Microphone Layout",
            "app/results/microphone_states",
            filter="CSV files (*.csv)",
        )[0]
        if layout_path:
            self.microphone_layout_path.setText(layout_path)

        self.mic_layout_path = layout_path

    def on_aero_pyramid_test_started(self):

        audio_file, _, _ = self.get_files()

        self.nspbufs = 10
        self.nfcsmps = 40
        self.naubufs = 5  # 5?

        speed_buffer_size = self.parent().control_widget.sample_buffer_size

        ntests = self.pyramid_steps * 2 - 1
        self.motor_data = np.zeros((ntests, self.nspbufs * speed_buffer_size, 4))
        self.force_data = np.zeros((ntests, self.nfcsmps, 3))
        # audio data is too fast the seperate thead writes it to a file

        # threads and their appropriate signals
        if self.parent().force_widget.serial_thread is None:
            return

        print("aero_pyramid Test Started")
        self.aero_pyramid_test_start_button.setEnabled(False)
        self.audio_pyramid_test_start_button.setEnabled(False)

        self.parent().control_widget.stop_button.clicked.connect(self.stop_aero_pyramid)

        self.parent().control_widget.controller.finishedLogging.connect(
            self.speed_callback
        )
        self.parent().control_widget.controller.finishedLogging.connect(
            self.check_aero_data_points
        )
        self.parent().force_widget.serial_thread.finishedLogging.connect(
            self.force_callback
        )
        self.parent().force_widget.serial_thread.finishedLogging.connect(
            self.check_aero_data_points
        )
        # self.parent().audio_widget.daq_thread.finishedLogging.connect(self.audio_callback) # pyqtSignal()
        # self.parent().audio_widget.daq_thread.finishedLogging.connect(self.check_aero_data_points)

        self.data_types_recieved = 0

        # self.parent().control_widget.controller.startLogging.emit(self.nspbufs) # pyqtSignal(nbuffers)
        # self.parent().force_widget.serial_thread.startLogging.emit(self.nfcsmps) # pyqtSignal(nsamples)
        # self.parent().audio_widget.daq_thread.startLogging.emit(audio_file,
        # self.naubufs) # pyqtSignal(file_name, nbuffers=None)

        self.parent().control_widget.start_control()

        self.aero_pyramid_step()
        # measure 10s acoustic and force
        # start speed aero_pyramid
        # each point measure speed, force
        # measure acoustic halfway up and at top of aero_pyramid

    def stop_aero_pyramid(self):
        self.parent().control_widget.controller.stopCheckingSettled.emit()
        self.on_aero_pyramid_test_finished()

    def speed_callback(self, data):
        print(f"Speed Callback, {data.shape}")
        self.motor_data[self.idx] = data

    def force_callback(self, data):
        print(f"Force Callback, {data.shape}")
        self.force_data[self.idx] = data

    def audio_callback(self):
        print(f"Audio Callback")

    def check_aero_data_points(self, _=None):
        self.data_types_recieved += 1

        if self.data_types_recieved == self.data_types_expected:
            self.data_types_recieved = 0
            self.aero_pyramid_step()

    def aerodynamic_collect(self):
        self.parent().control_widget.controller.speedSettled.disconnect(
            self.aerodynamic_collect
        )
        self.parent().control_widget.controller.startLogging.emit(
            self.nspbufs
        )  # pyqtSignal(nbuffers)
        self.parent().force_widget.serial_thread.startLogging.emit(
            self.nfcsmps
        )  # pyqtSignal(nsamples)
        self.data_types_expected = 2

    def aero_pyramid_step(self):
        # sets speed to next point in aero_pyramid

        if self.idx == 0:
            # first step
            pass
        elif self.idx == len(self.speed_pyramid) - 1:
            # last step
            self.step_timer.stop()
            self.on_aero_pyramid_test_finished()
            return

        speed = self.speed_pyramid[self.idx]
        self.parent().control_widget.speed_box.setText(str(speed))
        self.parent().control_widget.controller.setSpeed.emit(speed / 60)
        self.parent().control_widget.controller.speedSettled.connect(
            self.aerodynamic_collect
        )
        self.parent().control_widget.controller.startCheckingSettled.emit()

        self.idx += 1

    def on_aero_pyramid_test_finished(self):
        print("aero_pyramid Test Finished")
        self.parent().control_widget.stop_control()
        self.aero_pyramid_test_start_button.setEnabled(True)
        self.audio_pyramid_test_start_button.setEnabled(True)

        if self.idx == len(self.speed_pyramid) - 1:
            # save everything
            _, _, aero_file = self.get_files()
            np.savez(aero_file, force_data=self.force_data, motor_data=self.motor_data)

        self.idx = 0

        self.parent().control_widget.stop_button.clicked.disconnect(
            self.stop_aero_pyramid
        )

        self.parent().force_widget.serial_thread.finishedLogging.disconnect(
            self.force_callback
        )
        self.parent().force_widget.serial_thread.finishedLogging.disconnect(
            self.check_aero_data_points
        )
        self.parent().control_widget.controller.finishedLogging.disconnect(
            self.speed_callback
        )
        self.parent().control_widget.controller.finishedLogging.disconnect(
            self.check_aero_data_points
        )
        # self.parent().audio_widget.daq_thread.finishedLogging.disconnect(self.audio_callback)
        # self.parent().audio_widget.daq_thread.finishedLogging.disconnect(self.check_aero_data_points)

    def check_audio_data_points(self, _=None):
        self.data_types_recieved += 1

        if self.data_types_recieved == self.data_types_expected:
            self.data_types_recieved = 0
            # maybe append audio data to meta data
            self.audio_pyramid_step()

    def on_audio_pyramid_test_started(self):

        self.nspbufs = 10
        self.naubufs = 1 * self.parent().audio_widget.buffer_freq  # 1s
        speed_buffer_size = self.parent().control_widget.sample_buffer_size

        ntests = self.pyramid_steps * 2 - 1
        self.motor_data = np.zeros((ntests, self.nspbufs * speed_buffer_size, 4))
        self.audio_fnames = []
        # audio data is too fast the seperate thead writes it to a file

        # threads and their appropriate signals
        if self.parent().audio_widget.daq_thread is None:
            return

        print("audio pyramid Test Started")
        self.aero_pyramid_test_start_button.setEnabled(False)
        self.audio_pyramid_test_start_button.setEnabled(False)

        self.parent().control_widget.stop_button.clicked.connect(
            self.stop_audio_pyramid
        )

        self.parent().control_widget.controller.finishedLogging.connect(
            self.speed_callback
        )
        self.parent().control_widget.controller.finishedLogging.connect(
            self.check_audio_data_points
        )
        self.parent().audio_widget.daq_thread.finishedLogging.connect(
            self.audio_callback
        )  # pyqtSignal()
        self.parent().audio_widget.daq_thread.finishedLogging.connect(
            self.check_audio_data_points
        )

        self.data_types_recieved = 0

        # self.parent().control_widget.controller.startLogging.emit(self.nspbufs) # pyqtSignal(nbuffers)
        # self.parent().force_widget.serial_thread.startLogging.emit(self.nfcsmps) # pyqtSignal(nsamples)
        # self.parent().audio_widget.daq_thread.startLogging.emit(audio_file,
        # self.naubufs) # pyqtSignal(file_name, nbuffers=None)

        self.parent().control_widget.start_control()

        self.audio_pyramid_step()

    def audio_pyramid_step(self):
        # sets speed to next point in aero_pyramid

        if self.idx == 0:
            # first step
            pass
        elif self.idx == len(self.speed_pyramid) - 1:
            # last step
            self.step_timer.stop()
            self.on_audio_pyramid_test_finished()
            return

        speed = self.speed_pyramid[self.idx]
        self.parent().control_widget.speed_box.setText(str(speed))
        self.parent().control_widget.controller.setSpeed.emit(speed / 60)
        self.parent().control_widget.controller.speedSettled.connect(self.audio_collect)
        self.parent().control_widget.controller.startCheckingSettled.emit()

        print(f"Step {self.idx}, Speed: {speed}")

        self.idx += 1

    def audio_collect(self):
        self.parent().control_widget.controller.speedSettled.disconnect(
            self.audio_collect
        )
        self.parent().control_widget.controller.startLogging.emit(
            self.nspbufs
        )  # pyqtSignal(nbuffers)
        # self.parent().force_widget.serial_thread.startLogging.emit(self.nfcsmps)
        # # pyqtSignal(nsamples)
        audio_file, _, _ = self.get_files()
        audio_file = str(audio_file)
        self.audio_fnames.append(audio_file)

        buffer_freq = self.parent().audio_widget.buffer_freq
        sample_freq = self.parent().audio_widget.sample_freq
        duration = 1
        naubufs = np.ceil(duration * sample_freq / buffer_freq).astype(int)

        self.parent().audio_widget.daq_thread.startLogging.emit(audio_file, naubufs)

        self.data_types_expected = 2

    def on_audio_pyramid_test_finished(self):
        self.parent().control_widget.stop_control()
        self.aero_pyramid_test_start_button.setEnabled(True)
        self.audio_pyramid_test_start_button.setEnabled(True)

        # append all audio fnames and averaged motor data to meta data
        _, metaf, _ = self.get_files()
        for i, audio_file in enumerate(
            self.audio_fnames
        ):  # req: [speed, current, temp]
            # we have: [thetime, vel, current, temperature]
            _, speed, current, temp = np.mean(self.motor_data[i, :, :], axis=0)
            micpath = str(Path(self.microphone_layout_path.text()))
            append_audiof_to_metaf(audio_file, metaf, [speed, current, temp, micpath])

        self.idx = 0
        self.parent().control_widget.stop_button.clicked.disconnect(
            self.stop_audio_pyramid
        )

        self.parent().control_widget.controller.finishedLogging.disconnect(
            self.speed_callback
        )
        self.parent().control_widget.controller.finishedLogging.disconnect(
            self.check_audio_data_points
        )
        self.parent().audio_widget.daq_thread.finishedLogging.disconnect(
            self.audio_callback
        )
        self.parent().audio_widget.daq_thread.finishedLogging.disconnect(
            self.check_audio_data_points
        )

    def stop_audio_pyramid(self):
        self.parent().control_widget.controller.stopCheckingSettled.emit()

        # delete all of self.audio_fnames
        for audio_file in self.audio_fnames:
            if os.path.exists(audio_file):
                os.remove(audio_file)

        self.on_audio_pyramid_test_finished()


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.control_widget = ControlWidget(self)
        self.force_widget = ForceWidget(self)
        self.audio_widget = AudioWidget(self)
        self.test_widget = TestWidget(self)

        QApplication.instance().aboutToQuit.connect(self.about_to_quit)

        layout = QGridLayout()
        layout.addWidget(self.control_widget, 0, 0, 2, 1)
        layout.addWidget(self.test_widget, 0, 1, 1, 1)
        layout.addWidget(self.force_widget, 1, 1, 1, 1)
        layout.addWidget(self.audio_widget, 0, 2, 2, 1)

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
