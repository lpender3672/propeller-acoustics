
from PyQt6.QtWidgets import QVBoxLayout, QWidget, QGridLayout, QTabWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import numpy as np

from table import OutputTable, TableVar

from hanson import hanson_av
from betz import betz_off_design, bem

class PlotCanvas(FigureCanvas, QWidget):
    def __init__(self, parent=None, xlabel = "", ylabel = "", title = "", hideaxes = False):

        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        self.fig.tight_layout()

        FigureCanvas.__init__(self, self.fig)
        QWidget.__init__(self, parent)

        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title

        self.clear_lines()
        self.clear_points()
        self.clear_plot()

        if hideaxes:
            self.ax.axis('off')

    def add_lines(self, line_data, linestyle = None, label = None):
        if isinstance(line_data, list):
            line_data = np.array(line_data)

        self.line_data.append(line_data)

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
            n = len(self.line_labels)
            self.line_labels.append([str(n+i) for i in range(line_data.shape[0])])

        if len(self.line_data) > len(self.line_colors):
            self.line_data.pop(0)
            # keep colours consistent
            self.line_colors.append(self.line_colors[0])
            self.line_styles.append(self.line_styles[0])
            self.line_labels.append(self.line_labels[0])

            self.line_colors.pop(0)
            self.line_styles.pop(0)
            self.line_labels.pop(0)
        
        self.plot_data()
    
    def add_points(self, point_data, marker = None, label = None):
        if isinstance(point_data, list):
            point_data = np.array(point_data)

        self.point_data.append(point_data)

        if marker:
            if not isinstance(marker, list):
                marker = [marker] * (point_data.shape[0] - 1)
            assert len(marker) == point_data.shape[0] - 1
            self.point_markers.append(marker)
        else:
            self.point_markers.append(["o" for _ in range(point_data.shape[0])])

        if label:
            if not isinstance(label, list):
                label = [label]
            assert len(label) == point_data.shape[0] - 1, "Must have unique label"
            self.point_labels.append(label)
        else:
            n = len(self.point_labels)
            self.point_labels.append([str(n + i) for i in range(point_data.shape[0])])

        if len(self.point_data) > len(self.point_colors):
            self.point_data.pop(0)
            # keep colours consistent
            self.point_colors.append(self.point_colors[0])
            self.point_markers.append(self.point_markers[0])
            self.point_labels.append(self.point_labels[0])

            self.point_colors.pop(0)
            self.point_markers.pop(0)
            self.point_labels.pop(0)
        
        self.plot_data()
        
    def clear_lines(self):
        self.line_data = []
        self.line_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']
        self.line_styles = []
        self.line_labels = []
    
    def clear_points(self):
        self.point_data = []
        self.point_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']
        self.point_markers = []
        self.point_labels = []
    
    def clear_plot(self):
        self.ax.clear()
        self.ax.grid(True)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.ax.set_title(self.title)
        self.fig.tight_layout()
        self.draw()

    def plot_data(self):
        # clear

        for i, line_data in enumerate(self.line_data):

            for j in range(line_data.shape[0] - 1):

                self.ax.plot(
                    line_data[0], line_data[j+1],
                    linestyle= self.line_styles[i][j],
                    color= self.line_colors[i],
                    label= self.line_labels[i][j]
                )
        for i, point_data in enumerate(self.point_data):

            for j in range(point_data.shape[0] - 1):

                self.ax.plot(
                    point_data[0], point_data[j+1],
                    marker= self.point_markers[i][j],
                    color= self.point_colors[i],
                    label= self.point_labels[i][j]
                )

        self.set_ylim()

        self.ax.legend(loc='upper right')
        self.ax.grid(True)
        self.fig.tight_layout()
        self.draw()

    def set_ylim(self, bottom_override = None, top_override = None):
        
        miny = np.inf
        maxy = -np.inf
        for line_data in self.line_data:
            miny = min(miny, np.nanmin(line_data[1:]))
            maxy = max(maxy, np.nanmax(line_data[1:]))
        for point_data in self.point_data:
            miny = min(miny, np.nanmin(point_data[1:]))
            maxy = max(maxy, np.nanmax(point_data[1:]))
        
        if np.abs(miny) == np.inf:
            miny = np.sign(maxy)
        if np.abs(maxy) == np.inf:
            maxy = np.sign(miny)
        if bottom_override is not None:
            miny = bottom_override
        if top_override is not None:
            maxy = top_override

        drangey = 0.05 * (maxy - miny)
        if drangey == 0:
            drangey = 1e-3
        bottom = min(miny - drangey, miny + drangey)
        top = max(maxy - drangey, maxy + drangey)

        self.ax.set_ylim(
            bottom, top
        )


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
        
    def set_ylim(self):
        super().set_ylim(bottom_override=0)


class ResultsTable(OutputTable):
    def __init__(self, parent):
        
        vars = [
            TableVar(r"$C_T$", "[-]", "Thrust coefficient", float, 3),
            TableVar(r"$C_P$", "[-]", "Power coefficient", float, 3),
            TableVar(r"$FM$", "[-]", "Figure of Merit", float, 3),
            TableVar(r"$OASPL$", "[dB]", "Sound pressure level at observer", float, 3),

        ]
        self.keys = ["CT", "CP", "FM", "OASPL"]
        super().__init__(vars, parent)
    

    def set_values(self, res):
        for i, key in enumerate(self.keys):
            self.vars[i].value = res[key]
        super().set_values()
    
    def clear_values(self):
        for var in self.vars:
            var.value = None
        super().set_values()

    def update_results(self, avs):
        if avs.res['converged']:
            self.set_values(avs.res)
        else:
            self.clear_values()


class NoiseResultsWidget(QWidget):
    def __init__(self, parent, *args):
        super().__init__(parent,  *args)

        self.layout = QGridLayout(self)
        self.setLayout(self.layout)

        self.directivity = PolarPlotCanvas(self)
        self.directivity_toolbar = NavigationToolbar(self.directivity, self)

        self.thickness_interference = PlotCanvas(self, hideaxes=True)
        self.lift_interference = PlotCanvas(self, hideaxes=True)
        self.drag_interference = PlotCanvas(self, hideaxes=True)
        self.total_interference = PlotCanvas(self, hideaxes=True)

        self.layout.addWidget(self.directivity, 0, 0, 2, 2)
        self.layout.addWidget(self.directivity_toolbar)

        self.layout.addWidget(self.thickness_interference, 2, 0, 1, 1)
        self.layout.addWidget(self.lift_interference, 2, 1, 1, 1)
        self.layout.addWidget(self.drag_interference, 3, 0, 1, 1)
        self.layout.addWidget(self.total_interference, 3, 1, 1, 1)

        self.thickness_interference.line_colors = ['blue']
        self.lift_interference.line_colors = ['red']
        self.drag_interference.line_colors = ['green']
        self.total_interference.line_colors = ['black']

        self.directivity.setMinimumHeight(400)

    def update_results(self, avs):

        if not avs.res['converged']:
            return # no loading data if BEM not converged
        
        theta, V, L, D, theta_max, vector_contributions = hanson_av(avs)
        self.directivity.line_colors = ['blue', 'red']

        self.directivity.clear_plot()

        self.directivity.add_lines(np.array([theta, V, L, D]),
            linestyle=['-', ':', '-.'],
            label=['Thickness', 'Lift', 'Drag'])
        
        self.thickness_interference.clear_plot()
        self.lift_interference.clear_plot()
        self.drag_interference.clear_plot()
        self.total_interference.clear_plot()
        
        self.thickness_interference.add_lines(
            np.array([vector_contributions[0].real, vector_contributions[0].imag]),
            linestyle=['-'],
            label=['Thickness']
        )
        self.drag_interference.add_lines(
            np.array([vector_contributions[1].real, vector_contributions[1].imag]),
            linestyle=['-'],
            label=['Drag']
        )
        self.lift_interference.add_lines(
            np.array([vector_contributions[2].real, vector_contributions[2].imag]),
            linestyle=['-'],
            label=['Lift']
        )
        total = np.sum(vector_contributions, axis=0)
        self.total_interference.add_lines(
            np.array([total.real, total.imag]),
            linestyle=['-'],
            label=['Total']
        )

        avs.res['OASPL'] = 10 * np.log10(total[-1] * np.conj(total[-1])).real


class AerodynamicResultsWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        self.layout = QGridLayout(self)

        self.tabWidget = QTabWidget(self)

        self.profile = PlotCanvas(self, "$r/r_t$", "Sectional Coefficients ")
        #self.profile_toolbar = NavigationToolbar(self.profile, self)

        self.profile.line_colors = ['blue', 'red']

        self.performance = PlotCanvas(self, "$C_P$", "FM")
        #self.performance_toolbar = NavigationToolbar(self.performance, self)

        self.tabWidget.addTab(self.profile, "Profile")
        self.tabWidget.addTab(self.performance, "Performance")
        
        #self.layout.addWidget(self.profile, 0, 0, 2, 1)
        #self.layout.addWidget(self.profile_toolbar, 2, 0, 1, 1)
        #self.layout.addWidget(self.performance, 0, 1, 2, 1)
        #self.layout.addWidget(self.performance_toolbar, 2, 1, 1, 1)

        self.layout.addWidget(self.tabWidget)

    def update_results(self, avs):
        
        avs = betz_off_design(avs)
        #avs = bem(avs)
            # plot Cx and Cz against r0_rt
        
        if avs.res['converged']:
            self.profile.clear_plot()

            idxs = avs.res['invalids']

            # fill between xs
            if len(idxs) > 0:
                self.profile.ax.fill_betweenx(
                    [-1e3, 1e3],
                    avs.prop['r0_rt'][idxs[0]],
                    avs.prop['r0_rt'][idxs[-1]],
                    color='pink',
                    alpha=0.8,
                    label='Invalid'
                )

            self.profile.add_lines(
                [avs.prop['r0_rt'],
                avs.res['dCT'],
                avs.res['dCP']],
                linestyle=['--', ':'],
                label = ['$C_T$', '$C_P$']
            )

            self.performance.add_points(
                [avs.res['CP'],
                 avs.res['FM']])
        else:
            # indicate failed convergence
            self.profile.ax.text(0.5, 0.5, "Convergence Failed", fontsize=12, ha='center', va='center', transform=self.profile.ax.transAxes)
            self.profile.draw()
            


