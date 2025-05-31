import numpy as np
import sounddevice as sd

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QPushButton,
    QTabWidget,
    QWidget,
    QLabel,
)

from app.bem import (
    static_bem_swirl,
    static_bem_basic
)
from app.hanson import (
    calc_harmonics,
    get_radial_magnitudes,
    hanson,
    hanson_av,
    hanson_secondary_variables,
    sum_harmonics,
)
from app.table import (
    OutputTable, TableVar
)


class PlotCanvas(FigureCanvas, QWidget):
    def __init__(self, parent=None, xlabel="", ylabel="", title="", hideaxes=False):

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
            self.ax.axis("off")

        self.fig.tight_layout()

    def add_lines(self, line_data, linestyle=None, label=None):
        if isinstance(line_data, list):
            line_data = np.array(line_data)

        self.line_data.append(line_data)

        if linestyle:
            if not isinstance(linestyle, list):
                linestyle = [linestyle] * (line_data.shape[0] - 1)
            assert len(linestyle) == line_data.shape[0] - 1  # theta isnt styled
            self.line_styles.append(linestyle)
        else:
            self.line_styles.append(["-" for _ in range(line_data.shape[0])])

        if label:
            if not isinstance(label, list):
                label = [label]
            assert (
                len(label) == line_data.shape[0] - 1
            ), "Must have unique label"  # theta isnt labelled
            self.line_labels.append(label)
        else:
            n = len(self.line_labels)
            self.line_labels.append([str(n + i) for i in range(line_data.shape[0])])

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

    def add_points(self, point_data, marker=None, label=None):
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
        self.line_colors = [
            "blue",
            "red",
            "green",
            "orange",
            "purple",
            "brown",
            "pink",
            "gray",
            "cyan",
            "magenta",
        ]
        self.line_styles = []
        self.line_labels = []

    def clear_points(self):
        self.point_data = []
        self.point_colors = [
            "blue",
            "red",
            "green",
            "orange",
            "purple",
            "brown",
            "pink",
            "gray",
            "cyan",
            "magenta",
        ]
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

    def plot_data(self, draw=True):
        # clear

        for i, line_data in enumerate(self.line_data):

            for j in range(line_data.shape[0] - 1):

                self.ax.plot(
                    line_data[0],
                    line_data[j + 1],
                    linestyle=self.line_styles[i][j],
                    color=self.line_colors[i],
                    label=self.line_labels[i][j],
                )
        for i, point_data in enumerate(self.point_data):

            for j in range(point_data.shape[0] - 1):

                self.ax.plot(
                    point_data[0],
                    point_data[j + 1],
                    marker=self.point_markers[i][j],
                    color=self.point_colors[i],
                    label=self.point_labels[i][j],
                )

        self.set_ylim()

        self.ax.legend(loc="upper right")
        self.ax.grid(True)
        self.fig.tight_layout()

        if draw:
            self.draw()

    def set_ylim(self, bottom_override=None, top_override=None):

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

        self.ax.set_ylim(bottom, top)


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
        if self.ax.name == "polar":
            self.ax.set_theta_zero_location("N")
            self.ax.set_thetamin(0)
            self.ax.set_thetamax(180)

    def set_ylim(self):
        super().set_ylim(bottom_override=0)

    def plot_data(self):
        super().plot_data(draw=False)

        self.fig.subplots_adjust(right=0.7)
        self.ax.legend(loc="center right", bbox_to_anchor=(1.3, 0.5))

        self.draw()


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
        keys_to_set = [key for key in self.keys if key in res]

        for i, key in enumerate(keys_to_set):
            self.vars[i].value = res[key]
        super().set_values()

    def clear_values(self):
        for var in self.vars:
            var.value = None
        super().set_values()

    def update_results(self, avs):
        if avs.res["converged"]:
            self.set_values(avs.res)
        else:
            self.clear_values()


class AudioPlayerThread(QThread):
    finished = pyqtSignal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.audio_data = None

        # play 5s of audio
        self.sample_rate = 44100
        self.playback_rate = 1.0
        self.play_time = 1.0

    def run(self):
        if self.audio_data is not None:

            with sd.OutputStream(
                samplerate=44100, channels=1, dtype="float32"
            ) as stream:

                stream.write(self.audio_data)
                sd.sleep(int(self.play_time * 1000))
            
        self.finished.emit()

    def set_audio_data(self, audio_data):
        self.audio_data = audio_data
        if audio_data is not None:
            self.play_time = len(audio_data) / self.sample_rate


def synthesise(f0, harmonic_db, fs=44100, duration=1.0):

    n_samples = int(fs * duration)
    n_fft = n_samples

    harmonic_amp = 10 ** (harmonic_db / 20)

    spectrum = np.zeros(n_fft // 2 + 1, dtype=np.complex64)  # rfft size

    for i, amp in enumerate(harmonic_amp):
        freq = f0 * (i + 1)
        bin_index = int(np.round(freq / fs * n_fft))
        if bin_index < len(spectrum):

            phase = np.random.uniform(0, 2 * np.pi)
            spectrum[bin_index] += amp * np.exp(1j * phase)

    noise_level = 0.01  # noise magnitude
    spectrum += (
        np.random.randn(*spectrum.shape) + 1j * np.random.randn(*spectrum.shape)
    ) * noise_level

    signal = np.fft.irfft(spectrum, n=n_fft)
    signal /= np.max(np.abs(signal))
    return signal


class NoiseResultsWidget(QWidget):
    def __init__(self, parent, *args):
        super().__init__(parent, *args)

        self.layout = QGridLayout(self)
        self.setLayout(self.layout)

        self.directivity = PolarPlotCanvas(self)
        self.directivity_toolbar = NavigationToolbar(self.directivity, self)

        self.hmonic_plot = PlotCanvas(self, hideaxes=True)

        

        self.interference_tab = QTabWidget(self)
        interference_tab_widget = QWidget(self)
        interference_tab_layout = QGridLayout()

        self.harmonic_select = QComboBox(self)
        self.harmonic_select.addItems([str(i) for i in range(1, 30)])

        self.thickness_interference = PlotCanvas(interference_tab_widget, hideaxes=False)
        self.lift_interference = PlotCanvas(interference_tab_widget, hideaxes=False)
        self.drag_interference = PlotCanvas(interference_tab_widget, hideaxes=False)
        self.total_interference = PlotCanvas(interference_tab_widget, hideaxes=False)

        self.thickness_interference.line_colors = ["blue"]
        self.lift_interference.line_colors = ["red"]
        self.drag_interference.line_colors = ["green"]
        self.total_interference.line_colors = ["black"]
        self.hmonic_plot.line_colors = ["blue", "red"]
        self.directivity.line_colors = ["blue", "red"]

        interference_tab_layout.addWidget(self.thickness_interference, 0, 0)
        interference_tab_layout.addWidget(self.lift_interference, 1, 0)
        interference_tab_layout.addWidget(self.drag_interference, 1, 1)
        interference_tab_layout.addWidget(self.total_interference, 0, 1)

        
        interference_tab_widget.setLayout(interference_tab_layout)
        self.interference_tab.addTab(interference_tab_widget, "Interference")
        self.interference_tab.addTab(self.hmonic_plot, "Harmonics")

        self.directivity.setMinimumHeight(400)

        # audio play
        self.audio_player = AudioPlayerThread(self)
        self.audio_player.set_audio_data(None)

        self.play_audio_button = QPushButton("Play Audio", self)
        
        self.layout.addWidget(QLabel("Harmonic"), 0, 0, 1, 1)
        self.layout.addWidget(self.harmonic_select, 0, 1, 1, 1)
        self.layout.addWidget(self.play_audio_button, 3, 0, 1, 2)

        self.layout.addWidget(self.directivity, 1, 0, 2, 2)
        self.layout.addWidget(self.directivity_toolbar, 2, 0, 1, 2)
        self.layout.addWidget(self.interference_tab, 4, 0, 2, 2)

        self.harmonic_select.currentIndexChanged.connect(self.internal_update)
        self.play_audio_button.clicked.connect(self.on_audio_play)
        self.audio_player.finished.connect(self.on_audio_stop)

    def on_audio_play(self):
        # change name, remove old signal
        self.play_audio_button.setText("Stop Audio")
        self.play_audio_button.clicked.disconnect()
        self.play_audio_button.clicked.connect(self.on_audio_stop)
        self.audio_player.start()

    def on_audio_stop(self):
        self.play_audio_button.setText("Play Audio")
        self.play_audio_button.clicked.disconnect()
        self.play_audio_button.clicked.connect(self.on_audio_play)
        audio_data = self.audio_player.audio_data.copy()

        self.audio_player.terminate()
        self.audio_player = AudioPlayerThread(self)
        self.audio_player.set_audio_data(audio_data)
        self.audio_player.finished.connect(self.on_audio_stop)


    def internal_update(self):
        self.update_results(self.avs)

    def update_results(self, avs):

        if not avs.res["converged"]:
            return  # no loading data if BEM not converged

        self.avs = avs

        oper, prop, obs = hanson_secondary_variables(avs)
        theta = obs["theta"]

        ms = np.arange(1, 5)
        
        PVm, PDm, PLm = hanson(oper, prop, obs, ms, False)
        # V, L, D, total = sum_harmonics(PVm, PDm, PLm, avs.oper['pref'])

        peak_observer = {
            "r": [avs.oper["r_obs"] * avs.prop["rt"]],
            "theta": [avs.oper["theta_obs"]],
        }
        vector_contributions = get_radial_magnitudes(
            oper, prop, peak_observer, 1
        )  # TODO select m

        HVm, HDm, HLm = hanson(oper, prop, peak_observer, ms, False)
        # hmonics = calc_harmonics(HVm, HDm, HLm, avs.oper['pref'])

        f0 = oper["Omega"] * prop["B"] / (2 * np.pi)
        synthesised_signal = synthesise(f0, np.arange(1, 20), duration=1.0)
        self.audio_player.set_audio_data(synthesised_signal)

        self.directivity.clear_plot()

        idx = self.harmonic_select.currentIndex()

        self.directivity.add_lines(
            np.array([theta, PVm[:, idx], PDm[:, idx], PLm[:, idx]]),
            linestyle=["-", ":", "-."],
            label=["Thickness", "Lift", "Drag"],
        )

        self.thickness_interference.clear_plot()
        self.lift_interference.clear_plot()
        self.drag_interference.clear_plot()
        self.total_interference.clear_plot()
        self.hmonic_plot.clear_plot()

        self.thickness_interference.add_lines(
            np.array([vector_contributions[0].real, vector_contributions[0].imag]),
            linestyle=["-"],
            label=["Thickness"],
        )
        self.drag_interference.add_lines(
            np.array([vector_contributions[1].real, vector_contributions[1].imag]),
            linestyle=["-"],
            label=["Drag"],
        )
        self.lift_interference.add_lines(
            np.array([vector_contributions[2].real, vector_contributions[2].imag]),
            linestyle=["-"],
            label=["Lift"],
        )
        total = np.sum(vector_contributions, axis=0)
        self.total_interference.add_lines(
            np.array([total.real, total.imag]), linestyle=["-"], label=["Total"]
        )


"""
        self.hmonic_plot.add_lines(
            [ms, hmonics],
            linestyle=['-'],
            label=['Harmonics']
        )
"""
# avs.res['OASPL'] = 10 * np.log10(total[-1] * np.conj(total[-1])).real


class AerodynamicResultsWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        self.layout = QGridLayout(self)

        self.tabWidget = QTabWidget(self)

        self.CTprofile = PlotCanvas(self, "$r/r_t$ [-]", "$C_T/\\sigma$ [-]")
        self.CPprofile = PlotCanvas(self, "$r/r_t$ [-]", "$C_P$ [-]")

        self.CTprofile.line_colors = ["blue", "red"]
        self.CPprofile.line_colors = ["blue", "red"]

        self.performance = PlotCanvas(self, "$J$", "Coefficicent")
        self.AoA = PlotCanvas(self, "$r/r_t$ [-]", "$\\alpha$ [deg]")

        self.AoA.line_colors = ["blue", "red"]

        self.tabWidget.addTab(self.CTprofile, "Loading")
        self.tabWidget.addTab(self.CPprofile, "Power")
        #self.tabWidget.addTab(self.performance, "Performance")
        self.tabWidget.addTab(self.AoA, "Angle of Attack")

        # self.layout.addWidget(self.profile, 0, 0, 2, 1)
        # self.layout.addWidget(self.profile_toolbar, 2, 0, 1, 1)
        # self.layout.addWidget(self.performance, 0, 1, 2, 1)
        # self.layout.addWidget(self.performance_toolbar, 2, 1, 1, 1)

        self.layout.addWidget(self.tabWidget)

    def update_results(self, avs):

        # avs = betz_off_design(avs)
        # avs = bem(avs)
        avs = static_bem_basic(avs)
        # plot Cx and Cz against r0_rt

        if avs.res["converged"]:
            self.CTprofile.clear_plot()
            self.CPprofile.clear_plot()
            self.performance.clear_plot()
            self.AoA.clear_plot()

            #idxs = avs.res["invalids"]

            sigma = (
                avs.prop["B"]
                * avs.prop["c"]
                / (2 * np.pi * avs.prop["r0_rt"] * avs.prop["rt"])
            )

            self.CTprofile.add_lines(
                [avs.prop["r0_rt"], avs.res["dCT"]], linestyle=["--"]
            )

            self.CPprofile.add_lines(
                [avs.prop["r0_rt"], avs.res["dCQ"]], linestyle=["--"]
            )

            self.AoA.add_lines([avs.prop["r0_rt"], avs.res["alpha"]], linestyle=["--"])

        else:
            # indicate failed convergence
            # get active widget
            self.tabWidget.setCurrentIndex(0)
            self.CTprofile.ax.text(
                0.5,
                0.5,
                "Convergence Failed",
                fontsize=12,
                ha="center",
                va="center",
                transform=self.CTprofile.ax.transAxes,
            )
            self.CTprofile.draw()

        """
        Js = np.linspace(-0.1, 0.3, 10)
        Js, CPs, CTs, FMs = operating_range(avs, Js)
        self.performance.clear_plot()

        if CTs.size == 0:
            return

        self.performance.add_lines(
            [Js, CTs],
            linestyle=['-'],
            label=['CP']
        )
        """
