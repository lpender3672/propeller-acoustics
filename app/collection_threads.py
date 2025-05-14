from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
import pyqtgraph as pg
import sys
import datetime

import fibre
import nidaqmx
import nidaqmx.constants
import nidaqmx.error_codes
import nidaqmx.stream_readers
import nidaqmx.system
import nidaqmx.system.device
import numpy as np
import odrive
import serial
from nidaqmx.constants import (
    READ_ALL_AVAILABLE,
    AcquisitionType,
    LoggingMode,
    LoggingOperation,
)
from nidaqmx.stream_readers import AnalogMultiChannelReader
from nptdms import TdmsFile
from odrive.enums import *
from PyQt6.QtCore import QElapsedTimer, QObject, QThread, QTimer, pyqtSignal
from scipy.signal import butter, filtfilt


class SerialReaderThread(QThread):
    data_received = pyqtSignal(np.ndarray)
    error_occurred = pyqtSignal(Exception)

    startLogging = pyqtSignal(int)  # finite log only
    finishedLogging = pyqtSignal(np.ndarray)  # is emitted when a finite log finishes

    def __init__(self, serial_port, baud_rate):
        super().__init__()
        self.running = False
        self.serial = None
        try:
            self.serial = serial.Serial(serial_port, baud_rate, timeout=1)
            line = (
                self.serial.readline().decode("utf-8").strip().split(",")
            )  # current format in practical/pico/main.c
            assert len(line) == 3
        except serial.SerialException as e:
            self.error_occurred.emit(e)
            self.serial = None
            return
        except (AssertionError, ValueError) as e:
            self.error_occurred.emit(e)
            self.serial.close()
            self.serial = None
            return

        self.running = True
        self.sampling = False
        self.startLogging.connect(self.start_logging)

    def run(self):
        if not self.serial.is_open:
            self.serial.open()

        while self.running:
            try:
                line = self.serial.readline().decode("utf-8").strip()
                if line:
                    time_str, amp0_str, amp1_str = line.split(
                        ","
                    )  # current format in practical/pico/main.c
                    time_val = float(time_str.strip())
                    amp0_val = int(amp0_str.strip())
                    amp1_val = int(amp1_str.strip())
                    datapoint = np.array([time_val, amp0_val, amp1_val])

                    if self.sampling:
                        self.sample_data[self.sample_idx] = datapoint
                        self.sample_idx += 1
                        if self.sample_idx == self.sample_data.shape[0]:
                            self.sampling = False
                            self.finishedLogging.emit(self.sample_data)

                    self.data_received.emit(datapoint)
            except Exception as e:
                self.error_occurred.emit(e)
                # print(f"Error reading: pico be buggin {e}")

    def start_logging(self, nsamples):
        self.sampling = True
        self.sample_idx = 0
        self.sample_data = np.zeros((nsamples, 3))

    def stop(self):
        self.running = False
        self.serial.close()


class DAQThread(QThread):
    newSample = pyqtSignal(np.ndarray)
    errorOccurred = pyqtSignal(str)

    startLogging = pyqtSignal(str, int)  # finite or continous
    finishedLogging = pyqtSignal(str)  # is emitted when a finite log finishes
    stopLogging = pyqtSignal()  # finite or continous

    def __init__(self, parent=None, sample_rate=44000, group_samples=167, runtime=0):
        super().__init__(parent)

        try:
            self.task = nidaqmx.Task()
        except nidaqmx.errors.DaqNotFoundError:
            self.task = None

        self.total_channels = 0
        self.channel_names = []

        self.sample_rate = sample_rate
        self.group_samples = group_samples  # samples per rotation of the motor
        self.sample_buffer = np.zeros(
            (self.total_channels, self.group_samples), dtype=np.float64
        )
        self.runtime = runtime

        self.is_logging = False
        self.log_file = None
        self.max_buffers = None
        self.buffers_logged = 0

        self.startLogging.connect(self.start_logging)
        self.stopLogging.connect(self.stop_logging)

        self.timer = QElapsedTimer()

    def add_channel(self):

        if not self.task:
            return False

        channels_per_module = 4
        nmod = self.total_channels // channels_per_module
        nch = self.total_channels % channels_per_module
        chstr = f"cDAQ1Mod{nmod + 1}/ai{nch}"
        terminal_cfg = nidaqmx.constants.TerminalConfiguration.PSEUDO_DIFF

        try:
            self.task.ai_channels.add_ai_microphone_chan(
                chstr, terminal_config=terminal_cfg
            )
        except nidaqmx.DaqError as e:
            # if e.error_type == nidaqmx.error_codes.DAQmxErrors.DEV_CANNOT_BE_ACCESSED:
            print(e)
            return False

        self.channel_names.append(chstr)
        self.total_channels += 1
        self.sample_buffer = np.zeros(
            (self.total_channels, self.group_samples), dtype=np.float64
        )
        return True

    def run(self):

        # check if device attached
        if not self.task:
            return
        if not self.task.devices:
            return

        self.task.timing.cfg_samp_clk_timing(
            self.sample_rate,
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=self.group_samples,
        )
        print(f"ACTUAL FREQUENCY: {self.task.timing.samp_clk_rate}")
        self.task.register_every_n_samples_acquired_into_buffer_event(
            self.group_samples, self.callback
        )
        self.reader = AnalogMultiChannelReader(self.task.in_stream)

        if not np.isclose(self.task.timing.samp_clk_rate, self.sample_rate, atol=0.1):
            self.errorOccurred.emit(
                f"""Error setting sample rate: ensure requested rate is supported ({
                    self.task.timing.samp_clk_rate})"""
            )
            return

        self.task.start()
        self.timer.start()

        if self.runtime > 0:
            self.msleep(self.runtime * 1000)
            self.task.stop()

    def callback(
        self, task_handle, every_n_samples_event_type, number_of_samples, callback_data
    ):

        self.reader.read_many_sample(
            self.sample_buffer,
            self.group_samples,
        )
        self.newSample.emit(self.sample_buffer)

        if self.is_logging and self.log_file:

            self.sample_buffer.T.tofile(self.log_file)
            self.buffers_logged += 1
            if self.max_buffers is not None and self.buffers_logged > self.max_buffers:
                self.finishedLogging.emit(self.log_file.name)
                self.stop_logging()

        return 0

    def start_logging(self, file_name, nbuffers=None):
        self.log_file = open(file_name, "ab")  # binary append mode
        self.is_logging = True
        self.max_buffers = nbuffers
        self.buffers_logged = 0

    def stop_logging(self):
        self.is_logging = False
        if self.log_file:
            self.log_file.close()
            self.log_file = None
        self.max_buffers = None

    def stop(self):
        self.is_running = False
        if self.is_logging:
            self.stop_logging()
        if self.task:
            self.task.stop()
        self.wait()


class ControllerThread(QThread):
    setSpeed = pyqtSignal(float)
    startMotor = pyqtSignal()
    stopMotor = pyqtSignal()
    stateChanged = pyqtSignal(str)
    errorOccurred = pyqtSignal(str)
    newSample = pyqtSignal(np.ndarray)
    calibrateMotor = pyqtSignal()

    startLogging = pyqtSignal(int)  # finite or continous
    finishedLogging = pyqtSignal(np.ndarray)  # is emitted when a finite log finishes

    startCheckingSettled = pyqtSignal()
    stopCheckingSettled = pyqtSignal()
    speedSettled = pyqtSignal()

    def __init__(self, parent=None, sample_rate=200, buffer_size=10):
        super().__init__(parent)

        # timers removed as turned into an independent thread
        # self.watchdog_timer = QTimer(self)
        # self.watchdog_timer.timeout.connect(self.watchdog_loop)
        # self.aquisition_timer = QTimer(self)
        # self.aquisition_timer.timeout.connect(self.aquisition_loop)

        # velocity estimate, current control
        self.sample_rate = sample_rate
        self.dt_us = 1000000 / self.sample_rate
        self.buffer_size = buffer_size
        self.sample_idx = 0
        self.data_buffer = np.zeros((self.buffer_size, 4))

        self.log_buffers = 0
        self.logging = False

        self.checking_settled = False

        self.watchdog_dt = 0.1  # s
        self.samples_per_watchdog = int(self.watchdog_dt * self.sample_rate)

        self.odrv = None
        self.target_speed = 0.0
        self.thread_running = True  # thread running
        self.motor_running = False

        self.setSpeed.connect(self.set_speed)
        self.startMotor.connect(self.start_motor)
        self.stopMotor.connect(self.stop_motor)
        self.calibrateMotor.connect(self.calibrate_motor)

        self.timer = QElapsedTimer()  # time cannot be obtained from Odrive

        self.startLogging.connect(self.start_logging)
        self.startCheckingSettled.connect(self.start_checking_settled)
        self.stopCheckingSettled.connect(self.stop_checking_settled)

        self.last_watchdog_time = 0

        cutoff_freq = 10  # Hz
        order = 3
        nyquist = 0.5 * self.sample_rate  # Nyquist frequency
        normal_cutoff = cutoff_freq / nyquist  # Normalize cutoff frequency
        self.b, self.a = butter(order, normal_cutoff, btype="low", analog=False)

    def calibrate_motor(self):
        self.set_odrive_state(AxisState.FULL_CALIBRATION_SEQUENCE)

    def run(self):

        try:
            self.odrv = odrive.find_any(timeout=10)
            assert self.odrv
        except Exception as e:
            self.errorOccurred.emit(f"Error connecting to ODrive: {e}")
            self.thread_running = False
            return

        # controller settings
        self.odrv.axis0.controller.config.control_mode = ControlMode.VELOCITY_CONTROL
        self.odrv.axis0.controller.config.input_mode = InputMode.VEL_RAMP
        self.odrv.axis0.controller.config.vel_gain = 0.005
        self.odrv.axis0.controller.config.vel_integrator_gain = 0.001
        self.odrv.axis0.controller.config.vel_limit = 300
        self.odrv.axis0.controller.config.vel_ramp_rate = 100
        self.odrv.axis0.config.motor.current_soft_max = 30
        self.odrv.axis0.config.motor.current_hard_max = 38
        self.odrv.axis0.config.calibration_lockin.current = 20
        self.odrv.axis0.motor.motor_thermistor.config.temp_limit_lower = 80
        self.odrv.axis0.motor.motor_thermistor.config.temp_limit_upper = 100

        self.odrv.axis0.config.watchdog_timeout = 20 * self.watchdog_dt
        self.odrv.axis0.config.enable_watchdog = True
        # self.watchdog_timer.start(100)

        self.timer.start()

        loop_idx = 0

        while self.thread_running:

            loop_start_time = self.timer.nsecsElapsed()

            self.aquisition_loop()
            if loop_idx % self.samples_per_watchdog == 0:
                self.watchdog_loop()

            if self.motor_running:
                pass  # maybe do something

            loop_dt_us = (self.timer.nsecsElapsed() - loop_start_time) / 1000

            if loop_dt_us > self.dt_us:
                ratio = loop_dt_us / self.dt_us
                if ratio > 5:
                    pass
                    # self.errorOccurred.emit(f"Warning slow odrive communication {ratio:.2f}")
            else:
                self.usleep(int(self.dt_us - loop_dt_us))

            loop_idx += 1

    def stop_thread(self):
        self.stop_motor()
        self.thread_running = False
        self.exit()

    def watchdog_loop(self):

        # print(self.timer.elapsed() - self.last_watchdog_time)
        self.last_watchdog_time = self.timer.elapsed()

        try:
            motor_error = self.odrv.axis0.error
        except AttributeError:
            pass  # no error?
        else:
            self.errorOccurred.emit(motor_error)
            # self.stop()

        try:
            encoder_error = self.odrv.axis0.encoder.error
        except AttributeError:
            pass  # no error?
        else:
            self.errorOccurred.emit(encoder_error)
            # self.stop()

        # feed the beast
        self.odrv.axis0.watchdog_feed()

    def aquisition_loop(self):

        vel = 60 * self.odrv.axis0.vel_estimate

        try:
            current = self.odrv.axis0.motor.foc.Iq_measured
        except AttributeError:
            current = 0

        try:
            temperature = self.odrv.axis0.motor.motor_thermistor.temperature
        except AttributeError:
            temperature = 0

        thetime = self.timer.elapsed() / 1000
        self.data_buffer[self.sample_idx] = [thetime, vel, current, temperature]
        self.sample_idx += 1

        if self.sample_idx >= self.buffer_size:
            self.newSample.emit(self.data_buffer.copy())
            self.sample_idx = 0

            if self.checking_settled:
                fltrd = filtfilt(self.b, self.a, self.log_buffer[:, 1])
                if np.isclose(fltrd, -60 * self.target_speed, atol=5, rtol=0.025).all():
                    self.speedSettled.emit()
                    self.checking_settled = False
                    self.logging = False
                else:
                    pass

            if self.logging:
                self.log_buffer = np.roll(self.log_buffer, -self.buffer_size, axis=0)
                self.log_buffer[-self.buffer_size:] = self.data_buffer
                self.log_idx += 1

                if self.log_idx == self.log_buffers and not self.checking_settled:
                    self.logging = False
                    self.finishedLogging.emit(self.log_buffer)

    def start_logging(self, log_buffers=1):
        self.logging = True
        self.log_buffers = log_buffers
        self.log_buffer = np.zeros((log_buffers * self.buffer_size, 4))
        self.log_idx = 0

    def start_checking_settled(self):
        self.checking_settled = True
        self.start_logging(2)  # 2 buffers

    def stop_checking_settled(self):
        self.checking_settled = False
        self.logging = False

    def set_speed(self, speed):
        if not self.odrv:
            self.errorOccurred.emit("No ODrive connection. Cannot set speed.")
            return
        self.target_speed = float(speed)
        self.odrv.axis0.controller.input_vel = -self.target_speed

    def start_motor(self):
        if not self.odrv:
            self.errorOccurred.emit("No ODrive connection. Cannot start motor.")
            return
        try:
            self.set_odrive_state(AxisState.CLOSED_LOOP_CONTROL)
            self.odrv.axis0.controller.input_vel = -self.target_speed
            self.motor_running = True
        except Exception as e:
            self.errorOccurred.emit(f"Error starting motor: {e}")

    def stop_motor(self):
        if not self.odrv:
            self.errorOccurred.emit("No ODrive connection. Cannot stop motor.")
            return
        try:
            self.motor_running = False
            self.odrv.axis0.controller.input_vel = 0.0
            self.set_odrive_state(AxisState.IDLE)
        except Exception as e:
            self.errorOccurred.emit(f"Error stopping motor: {e}")

    def set_odrive_state(self, state):
        if not self.odrv:
            self.errorOccurred.emit("No ODrive connection.")
            return
        try:
            self.odrv.axis0.requested_state = state
            self.stateChanged.emit(f"ODrive state changed to {state}")
        except Exception as e:
            self.errorOccurred.emit(f"Failed to set ODrive state: {e}")

    def reset_config(self):
        print("Erasing previous configuration")

        try:
            self.odrv.erase_configuration()
        except fibre.libfibre.ObjectLostError:
            pass

        # wait for the odrive to reboot
        self.msleep(5000)

        odrv = odrive.find_any()

        odrv.config.dc_bus_overvoltage_trip_level = 15
        odrv.config.dc_bus_undervoltage_trip_level = 10.5
        odrv.config.dc_max_positive_current = 10
        odrv.config.dc_max_negative_current = -np.inf
        odrv.config.brake_resistor0.enable = False
        odrv.axis0.config.motor.motor_type = MotorType.HIGH_CURRENT
        odrv.axis0.config.motor.pole_pairs = 7
        odrv.axis0.motor.motor_thermistor.config.r_ref = 10000
        odrv.axis0.motor.motor_thermistor.config.beta = 3950
        odrv.axis0.motor.motor_thermistor.config.temp_limit_lower = 280
        odrv.axis0.motor.motor_thermistor.config.temp_limit_upper = 300
        odrv.axis0.config.motor.torque_constant = 0.0035956521739130432
        odrv.axis0.config.motor.current_soft_max = 10
        odrv.axis0.config.motor.current_hard_max = 15
        odrv.axis0.config.motor.calibration_current = 10
        odrv.axis0.config.motor.resistance_calib_max_voltage = 2
        odrv.axis0.config.calibration_lockin.current = 10
        odrv.axis0.motor.motor_thermistor.config.enabled = True

        odrv.axis0.config.torque_soft_min = -np.inf
        odrv.axis0.config.torque_soft_max = np.inf
        odrv.axis0.trap_traj.config.accel_limit = 5
        odrv.can.config.protocol = Protocol.NONE
        odrv.axis0.config.enable_watchdog = False
        odrv.config.enable_uart_a = False
        odrv.axis0.config.load_encoder = EncoderId.SPI_ENCODER0
        odrv.axis0.config.commutation_encoder = EncoderId.SPI_ENCODER0
        odrv.spi_encoder0.config.mode = SpiEncoderMode.AMS
        odrv.spi_encoder0.config.ncs_gpio = 12

        print("Applying new configuration")

        try:
            odrv.save_configuration()
            odrv.reboot()
        except fibre.libfibre.ObjectLostError:
            pass

        # now wait for the odrive to reboot
        self.msleep(2000)

        del odrv

        self.odrv = odrive.find_any()


class CollectionTestWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Real-Time DAQ Plotter")
        self.resize(800, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("w")
        self.plot_widget.setTitle("Real-Time Voltage Data", color="b", size="12pt")
        self.plot_widget.setLabel("left", "Voltage", units="V")
        self.plot_widget.setLabel("bottom", "Sample Index")
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_curve = self.plot_widget.plot(pen=pg.mkPen(color="r", width=2))

        layout.addWidget(self.plot_widget)

        self.data_buffer = np.zeros(4400)
        self.daq_thread = DAQThread(sample_rate=44000, group_samples=4400)
        self.daq_thread.newSample.connect(self.update_plot)

        self.daq_thread.add_channel()
        self.daq_thread.start()

    def update_plot(self, data):
        if data.shape[0] > 0:
            channel_data = data[0]
            self.data_buffer = np.roll(self.data_buffer, -len(channel_data))
            self.data_buffer[-len(channel_data):] = channel_data
            self.plot_curve.setData(self.data_buffer)

    def closeEvent(self, event):
        self.daq_thread.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = CollectionTestWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
