
import nidaqmx.error_codes
import nidaqmx.stream_readers
import serial
from PyQt6.QtCore import QThread, pyqtSignal, QObject, QTimer
from odrive.enums import *
import odrive

import datetime

from nptdms import TdmsFile
import nidaqmx
from nidaqmx.stream_readers import AnalogMultiChannelReader

from nidaqmx.constants import (
    READ_ALL_AVAILABLE,
    AcquisitionType,
    LoggingMode,
    LoggingOperation,
)



import numpy as np
import fibre

class SerialReaderThread(QThread):
    data_received = pyqtSignal(float, int, int)

    def __init__(self, serial_port, baud_rate):
        super().__init__()
        self.serial = serial.Serial(serial_port, baud_rate, timeout=1)
        self.running = True

    def run(self):
        if not self.serial.is_open:
            self.serial.open()

        while self.running:
            try:
                line = self.serial.readline().decode('utf-8').strip()
                if line:
                    time_str, amp0_str, amp1_str = line.split(",") # current format in practical/pico/main.c
                    time_val = float(time_str.strip())
                    amp0_val = int(amp0_str.strip())
                    amp1_val = int(amp1_str.strip())
                    self.data_received.emit(time_val, amp0_val, amp1_val)
            except Exception as e:
                print(f"Error reading: pico be buggin {e}")

    def stop(self):
        self.running = False
        self.serial.close()

class SerialThreadWrapper(QObject):
    # simply saves and parses the data from the serial thread
    data_received = pyqtSignal(np.ndarray)
    
    def __init__(self, serial_port, baud_rate, samples):
        super().__init__()
        self.serial_thread = SerialReaderThread(serial_port, baud_rate)
        
        self.samples = samples
        self.sample_idx = 0
        self.data = np.zeros((samples, 3), dtype=np.float64)

        self.serial_thread.data_received.connect(self.append_data)

        self.stop_timer = QTimer(is_single_shot=True)
        self.stop_timer.timeout.connect(self.serial_thread.stop)

        self.serial_thread.start()
        self.stop_timer.start(samples * 1000 / 80) # Hx711 have 80 Hz sampling rate
        
    def append_data(self, time, amp0, amp1):
        self.data[self.sample_idx] = [time, amp0, amp1]
        self.sample_idx += 1

        if self.sample_idx == self.samples:
            self.data_received.emit(self.data)
            self.serial_thread.stop()

class DAQThread(QThread):
    newSample = pyqtSignal(np.ndarray)
    errorOccurred = pyqtSignal(str)

    startLogging = pyqtSignal(str, int) # finite or continous
    finishedLogging = pyqtSignal() # is emitted when a finite log finishes
    stopLogging = pyqtSignal() # finite or continous

    def __init__(self, parent=None, sample_rate=44000, group_samples=167, runtime=0):
        super().__init__(parent)

        self.task = nidaqmx.Task()
        self.total_channels = 0

        self.sample_rate = sample_rate
        self.group_samples = group_samples # samples per rotation of the motor
        self.sample_buffer = np.zeros((self.total_channels, self.group_samples), dtype=np.float64)
        self.runtime = runtime

        self.is_logging = False
        self.log_file = None
        self.max_buffers = None
        self.buffers_logged = 0

    def add_channel(self):
        self.total_channels += 1

        channels_per_module = 4
        nmod = self.total_channels // channels_per_module
        nch = self.total_channels % channels_per_module
        chstr = f"cDAQ1Mod{nmod+1}/ai{nch}"

        try:
            self.task.ai_channels.add_ai_voltage_chan(chstr)
        except nidaqmx.DaqError as e:
            #if e.error_type == nidaqmx.error_codes.DAQmxErrors.DEV_CANNOT_BE_ACCESSED:
            return False

        self.sample_buffer = np.zeros((self.total_channels, self.group_samples), dtype=np.float64)
        return True

    def remove_channel(self):
        if self.total_channels > 0:
            self.total_channels -= 1

            channels_per_module = 4
            nmod = self.total_channels // channels_per_module
            nch = self.total_channels % channels_per_module
            chstr = f"cDAQ1Mod{nmod+1}/ai{nch}"

            self.task.ai_channels.remove_ai_voltage_chan(chstr)
            self.sample_buffer = np.zeros((self.total_channels, self.group_samples), dtype=np.float64)

    def run(self):

        self.task.timing.cfg_samp_clk_timing(
            self.sample_rate,
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=self.group_samples
        )
        self.task.register_every_n_samples_acquired_into_buffer_event(100, self.callback)
        self.reader = AnalogMultiChannelReader(self.task.in_stream)

        self.task.start()

        if self.runtime > 0:
            self.msleep(self.runtime * 1000)
            self.task.stop()
        
    def callback(self, task_handle, every_n_samples_event_type, number_of_samples, callback_data):
        
        self.reader.read_many_sample(
            self.sample_buffer,
            self.group_samples,
        )
        self.newSample.emit(self.sample_buffer)

        if self.is_logging and self.log_file:
                self.sample_buffer.T.tofile(self.log_file)
                self.buffers_logged += 1
                if self.max_buffers is not None and self.buffers_logged >= self.max_buffers:
                    self.stop_logging()
        return 0

    def start_logging(self, file_name, max_buffers=None):
        self.log_file = open(file_name, "ab")  # binary append mode
        self.is_logging = True
        self.max_buffers = max_buffers
        self.buffers_logged = 0

    def stop_logging(self):
        self.is_logging = False
        if self.log_file:
            self.log_file.close()
            self.log_file = None
        self.max_buffers = None

    def stop(self):
        self.is_running = False
        self.stop_logging()
        if self.task:
            self.task.stop()
        self.wait()


class Controller(QObject):
    speedChanged = pyqtSignal(float)
    stateChanged = pyqtSignal(str)
    errorOccurred = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.control_loop)

        self.odrv = None
        self.current_speed = 0.0
        self.running = False

        try:
            self.odrv = odrive.find_any()
            assert self.odrv
        except Exception as e:
            self.errorOccurred.emit(f"Error connecting to ODrive: {e}")

    def start(self):
        if not self.odrv:
            self.errorOccurred.emit("No ODrive connection. Cannot start control loop.")
            return

        self.running = True
        self.odrv.axis0.config.watchdog_timeout = 0.5
        self.odrv.axis0.config.enable_watchdog = True
        self.timer.start(100) # 200 ms

    def stop(self):
        self.running = False
        self.timer.stop()
        self.set_odrive_state(AxisState.AXIS_STATE_IDLE)

    def control_loop(self):

        self.odrv.axis0.watchdog_feed()

        motor_error = self.odrv.axis0.error
        encoder_error = self.odrv.axis0.encoder.error

        if motor_error or encoder_error:
            self.stop()

        if not self.running or not self.odrv:
            return

        try:
            self.odrv.axis0.controller.input_vel = self.current_speed

        except Exception as e:
            self.errorOccurred.emit(f"Error in control loop: {e}")
        

    def set_speed(self, speed):
        self.current_speed = speed
        self.speedChanged.emit(speed)

    def start_motor(self):
        if not self.odrv:
            self.errorOccurred.emit("No ODrive connection. Cannot start motor.")
            return

        try:
            self.set_odrive_state(AxisState.AXIS_STATE_CLOSED_LOOP_CONTROL)
            self.running = True
        except Exception as e:
            self.errorOccurred.emit(f"Error starting motor: {e}")

    def stop_motor(self):
        if not self.odrv:
            self.errorOccurred.emit("No ODrive connection. Cannot stop motor.")
            return

        try:
            self.running = False
            self.odrv.axis0.controller.input_vel = 0.0
            self.set_odrive_state(AxisState.AXIS_STATE_IDLE)
            print("Motor stopped.")
        except Exception as e:
            print(f"Error stopping motor: {e}")

    def set_odrive_state(self, state):
        if not self.odrv:
            self.errorOccurred.emit("No ODrive connection.")
            return

        try:
            self.odrv.axis0.requested_state = state
            self.stateChanged.emit(f"ODrive state changed to {state}")
        except Exception as e:
            self.errorOccurred.emit(f"Failed to set ODrive state: {e}")
            self.running = False

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
        odrv.axis0.config.motor.torque_constant = 0.0035956521739130432
        odrv.axis0.config.motor.current_soft_max = 10
        odrv.axis0.config.motor.current_hard_max = 15
        odrv.axis0.config.motor.calibration_current = 10
        odrv.axis0.config.motor.resistance_calib_max_voltage = 2
        odrv.axis0.config.calibration_lockin.current = 10
        odrv.axis0.motor.motor_thermistor.config.enabled = False

        odrv.axis0.config.torque_soft_min = -np.inf
        odrv.axis0.config.torque_soft_max = np.inf
        odrv.axis0.trap_traj.config.accel_limit = 5
        odrv.can.config.protocol = Protocol.NONE
        odrv.axis0.config.enable_watchdog = False
        odrv.config.enable_uart_a = False

        ## sensorless control

        odrv.axis0.controller.config.vel_limit = 358
        odrv.axis0.config.sensorless_ramp.vel = 200
        odrv.axis0.config.sensorless_ramp.accel = 20
        odrv.axis0.config.sensorless_ramp.current = 10
        odrv.axis0.controller.config.vel_gain = 0.001
        odrv.axis0.controller.config.vel_integrator_gain = 0.005
        odrv.axis0.config.load_encoder = EncoderId.SENSORLESS_ESTIMATOR
        odrv.axis0.config.commutation_encoder = EncoderId.SENSORLESS_ESTIMATOR

        odrv.axis0.config.sensorless_ramp.current = odrv.axis0.config.motor.current_soft_max

        # set  ramped velocity input mode
        #odrv0.axis0.controller.config.vel_ramp_rate = 0.5
        odrv.axis0.controller.config.control_mode = ControlMode.VELOCITY_CONTROL
        odrv.axis0.controller.config.input_mode = InputMode.VEL_RAMP

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


from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
import pyqtgraph as pg
import sys

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