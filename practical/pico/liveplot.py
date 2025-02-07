import sys
import serial
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt6.QtCore import QTimer, QThread, pyqtSignal, QObject
import pyqtgraph as pg


class SerialReaderThread(QThread):
    data_received = pyqtSignal(float, int, int)

    def __init__(self, serial_port, baud_rate):
        super().__init__()
        self.serial = serial.Serial(serial_port, baud_rate, timeout=1)
        self.running = True

    def run(self):
        while self.running:
            try:
                line = self.serial.readline().decode('utf-8').strip()
                if line:
                    time_str, amp0_str, amp1_str = line.split(",")
                    time_val = float(time_str.strip())
                    amp0_val = int(amp0_str.strip())
                    amp1_val = int(amp1_str.strip())
                    self.data_received.emit(time_val, amp0_val, amp1_val)
            except Exception as e:
                print(f"Error reading: pico be buggin {e}")

    def stop(self):
        self.running = False
        self.serial.close()


class SerialPlotter(QMainWindow):
    def __init__(self, serial_port, baud_rate=115200):
        super().__init__()

        self.plot_widget_amp0 = pg.PlotWidget()
        self.plot_widget_amp1 = pg.PlotWidget()
        self.plot_widget_amp0.setTitle("amp 0 Data")
        self.plot_widget_amp1.setTitle("amp 1 Data")

        central_widget = QWidget()
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        layout.addWidget(self.plot_widget_amp0)
        layout.addWidget(self.plot_widget_amp1)
        self.setCentralWidget(central_widget)

        self.time_buffer = []
        self.amp0_buffer = []
        self.amp1_buffer = []

        self.amp0_curve = self.plot_widget_amp0.plot(pen='r', name="amp 0")
        self.amp1_curve = self.plot_widget_amp1.plot(pen='b', name="amp 1")

        self.serial_thread = SerialReaderThread(serial_port, baud_rate)
        self.serial_thread.data_received.connect(self.add_data)
        self.serial_thread.start()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(100)

        self.max_buffer_size = 80 * 60 # 80 samples per second for 1 minute

    def add_data(self, time_val, amp0_val, amp1_val):
        self.time_buffer.append(time_val)
        self.amp0_buffer.append(amp0_val)
        self.amp1_buffer.append(amp1_val)

        if len(self.time_buffer) > self.max_buffer_size:
            self.time_buffer = self.time_buffer[-self.max_buffer_size:]
            self.amp0_buffer = self.amp0_buffer[-self.max_buffer_size:]
            self.amp1_buffer = self.amp1_buffer[-self.max_buffer_size:]

    def update_plot(self):
        if len(self.time_buffer) == len(self.amp0_buffer):
            self.amp0_curve.setData(self.time_buffer, self.amp0_buffer)

        if len(self.time_buffer) == len(self.amp1_buffer):
            self.amp1_curve.setData(self.time_buffer, self.amp1_buffer)

    def closeEvent(self, event):
        self.serial_thread.stop()
        self.serial_thread.wait()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = SerialPlotter(serial_port='COM3', baud_rate=115200)
    window.setWindowTitle("Serial Data Plotter")
    window.resize(800, 600)
    window.show()

    sys.exit(app.exec())
