
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import numpy as np


class PolarPlotCanvas(FigureCanvas, QWidget):
    def __init__(self, parent=None):

        self.fig = Figure()
        self.ax = self.fig.add_subplot(111, polar=True)
        self.fig.tight_layout()

        FigureCanvas.__init__(self, self.fig)
        QWidget.__init__(self, parent)

        self.clear_lines()

    def add_line(self, line_data, linestyle = None, label = None):
        self.data.append(line_data)

        if linestyle:
            if not isinstance(linestyle, list):
                linestyle = [linestyle]
            assert len(linestyle) == line_data.shape[0] - 1 # theta isnt styled
            self.line_styles.append(linestyle)
        else:
            self.line_styles.append(["-" for _ in range(line_data.shape[0])])

        if label:
            if not isinstance(label, list):
                label = [label]
            assert len(label) == line_data.shape[0] - 1 # theta isnt labelled
            self.line_labels.append(label)
        else:
            self.line_labels.append([str(i) for i in range(line_data.shape[0])])

        if len(self.data) > len(self.line_colors):
            self.data.pop(0)
            # keep colours consistent
            self.line_colors.append(self.line_colors[0])
            self.line_styles.append(self.line_styles[0])
            self.line_labels.append(self.line_labels[0])

            self.line_colors.pop(0)
            self.line_styles.pop(0)
            self.line_labels.pop(0)
        
        self.plot_data()

    def clear_lines(self):
        self.data = []
        self.line_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']
        self.line_styles = []
        self.line_labels = []

    def plot_data(self):
        # clear
        self.ax.clear()

        for i, line_data in enumerate(self.data):
            for j in range(line_data.shape[0] - 1):

                self.ax.plot(
                    line_data[0], line_data[j+1],
                    linestyle= self.line_styles[i][j],
                    color= self.line_colors[i],
                    label= self.line_labels[i][j]
                )
        

        self.ax.set_title('Polar Plot Example', va='bottom')
        self.ax.legend(loc='upper right')
        self.ax.grid(True)

        self.draw()


class NoiseResultsWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        self.layout = QVBoxLayout(self)
        self.setLayout(self.layout)

        self.canvas = PolarPlotCanvas(self)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.layout.addWidget(self.canvas)
        self.layout.addWidget(self.toolbar)


