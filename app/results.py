
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import numpy as np

from hanson import hanson_av
from betz import betz_off_design

class PlotCanvas(FigureCanvas, QWidget):
    def __init__(self, parent=None):

        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        self.fig.tight_layout()

        FigureCanvas.__init__(self, self.fig)
        QWidget.__init__(self, parent)

        self.clear_lines()
        self.clear_plot()

    def add_line(self, line_data, linestyle = None, label = None):
        self.data.append(line_data)

        if linestyle:
            if not isinstance(linestyle, list):
                linestyle = [linestyle] * (line_data.shape[0] - 1)
            assert len(linestyle) == line_data.shape[0] - 1 # theta isnt styled
            self.line_styles.append(linestyle)
        else:
            self.line_styles.append(["-" for _ in range(line_data.shape[0])])

        if label:
            if not isinstance(label, list):
                label = [label]
            assert len(label) == line_data.shape[0] - 1, "Must have unique label" # theta isnt labelled
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
    
    def clear_plot(self):
        self.ax.clear()

    def plot_data(self):
        # clear
        self.clear_plot()


        for i, line_data in enumerate(self.data):

            for j in range(line_data.shape[0] - 1):

                self.ax.plot(
                    line_data[0], line_data[j+1],
                    linestyle= self.line_styles[i][j],
                    color= self.line_colors[i],
                    label= self.line_labels[i][j]
                )
        
        miny = np.inf
        maxy = -np.inf
        for line_data in self.data:
            miny = min(miny, np.nanmin(line_data[1:]))
            maxy = max(maxy, np.nanmax(line_data[1:]))
        miny = 0
        self.ax.set_ylim( miny, maxy + 5)

        self.ax.set_title('Polar Plot Example', va='bottom')
        self.ax.legend(loc='upper right')
        self.ax.grid(True)

        self.draw()


class PolarPlotCanvas(PlotCanvas):
    def __init__(self, parent=None):
        super().__init__(parent)

        if self.ax:
            self.fig.delaxes(self.ax)
        
        self.ax = self.fig.add_subplot(111, polar=True)

        self.clear_lines()
        self.clear_plot()

    def clear_plot(self):
        super().clear_plot()
        if self.ax.name == 'polar':
            self.ax.set_theta_zero_location('N')
            self.ax.set_thetamin(0)
            self.ax.set_thetamax(180)


class NoiseResultsWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        self.layout = QVBoxLayout(self)
        self.setLayout(self.layout)

        self.canvas = PolarPlotCanvas(self)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.layout.addWidget(self.canvas)
        self.layout.addWidget(self.toolbar)

    def update_results(self, avs):
        
        data = hanson_av(avs)
        self.canvas.line_colors = ['blue', 'red']

        self.canvas.add_line(np.array(data),
            linestyle=['-', ':', '-.'],
            label=['Thickness', 'Lift', 'Drag'])


class AerodynamicResultsWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        self.layout = QVBoxLayout(self)
        self.setLayout(self.layout)

        self.canvas = PlotCanvas(self)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.layout.addWidget(self.canvas)
        self.layout.addWidget(self.toolbar)

    def update_results(self, avs):
        
        betz_off_design(avs)
        


