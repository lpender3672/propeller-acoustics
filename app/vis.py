from PyQt6.QtWidgets import (
    QVBoxLayout,
    QWidget,
    QGridLayout,
    QCheckBox,
    QFileDialog,
    QPushButton,
    QMessageBox,
    QDialog,
)
from PyQt6.QtCore import Qt, pyqtSignal
import pyqtgraph.opengl as gl
import numpy as np
from stl import mesh
from matplotlib import cm


class STLViewerDialog(QDialog):
    # when widget is double clicked, open dialog
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.widget = STLViewerWidget(self, popup=True)
        layout.addWidget(self.widget)

        self.setMinimumSize(800, 600)
        self.setWindowFlags(Qt.WindowType.Window)

        self.widget.view.escapePressed.connect(self.close)


class EventGLViewWidget(gl.GLViewWidget):
    doubleClicked = pyqtSignal()
    escapePressed = pyqtSignal()

    def mouseDoubleClickEvent(self, event):
        self.doubleClicked.emit()
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.escapePressed.emit()
        super().keyPressEvent(event)


def create_arrow(start, end, color=(1, 0, 0, 1), rfac=0.01):
    # Create a line (shaft)
    line_points = np.array([start, end])
    line = gl.GLLinePlotItem(pos=line_points, color=color, width=2, antialias=True)

    # Create an arrowhead (cone)
    arrow_direction = np.array(end, dtype=float) - np.array(start, dtype=float)
    arrow_length = np.linalg.norm(arrow_direction)

    cylradius = rfac * arrow_length
    coneradius = 2 * cylradius

    z_axis = np.array([0, 0, 1])  # Default cone orientation along z-axis
    rotation_vector = np.cross(z_axis, arrow_direction)
    rotation_angle = np.arccos(np.dot(z_axis, arrow_direction)) * 180 / np.pi

    if np.linalg.norm(rotation_vector) > 0:
        rotation_vector /= np.linalg.norm(rotation_vector)
    else:
        rotation_vector = z_axis

    cylinder_meshdata = gl.MeshData.cylinder(
        rows=10, cols=20, radius=[cylradius, cylradius], length=arrow_length
    )
    cylinder_mesh = gl.GLMeshItem(
        meshdata=cylinder_meshdata, smooth=True, color=color, shader="shaded"
    )

    # Create a cone mesh for the arrowhead
    cone_meshdata = gl.MeshData.cylinder(
        rows=10, cols=20, radius=[coneradius, 0], length=coneradius * 2
    )
    cone_mesh = gl.GLMeshItem(
        meshdata=cone_meshdata, smooth=True, color=color, shader="shaded"
    )

    # Apply the rotation matrix to the cylinder and cone
    cylinder_mesh.rotate(rotation_angle, *rotation_vector[0:3])
    cone_mesh.rotate(rotation_angle, *rotation_vector[0:3])

    cylinder_mesh.translate(*start)
    cone_mesh.translate(*end)

    return cylinder_mesh, cone_mesh


class STLViewerWidget(QWidget):
    def __init__(self, parent=None, popup=False):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.view = EventGLViewWidget()
        self.popup = popup
        self.dialog = None

        viewSettingWidget = QWidget()
        viewSettingLayout = QGridLayout(viewSettingWidget)

        self.smoothToggle = QCheckBox("Smooth")
        self.smoothToggle.setChecked(False)
        self.smoothToggle.stateChanged.connect(self.update_mesh_plot)

        self.edgeToggle = QCheckBox("Draw Edges")
        self.edgeToggle.setChecked(False)
        self.edgeToggle.stateChanged.connect(self.update_mesh_plot)

        self.arrowToggle = QCheckBox("Show Arrow")
        self.arrowToggle.setChecked(False)
        self.arrowToggle.stateChanged.connect(self.update_mesh_plot)

        self.save_button = QPushButton("Save to .stl")
        self.save_button.clicked.connect(self.save_stl_file)

        self.view.doubleClicked.connect(self.display_fullscreen_dialog)

        # viewSettingLayout.addWidget(self.smoothToggle, 0, 0)
        viewSettingLayout.addWidget(self.arrowToggle, 0, 0)
        viewSettingLayout.addWidget(self.edgeToggle, 0, 1)

        if not popup:
            viewSettingLayout.addWidget(self.save_button, 0, 2)

        viewSettingWidget.setMaximumHeight(50)
        layout.addWidget(viewSettingWidget)
        layout.addWidget(self.view)

        self.view.setCameraPosition(distance=1)

        self.num_blades = 1

    def display_fullscreen_dialog(self):
        if self.popup:
            return
        if self.dialog:
            self.dialog.close()
            self.dialog = None
            return

        self.dialog = STLViewerDialog(self)
        self.dialog.widget.set_mesh(self.stl_mesh)
        self.dialog.show()

    def load_stl_file(self, stl_file):
        self.stl_file = stl_file
        self.stl_mesh = mesh.Mesh.from_file(stl_file)
        self.update_mesh_plot()

    def save_stl_file(self, stl_file):
        path = QFileDialog.getSaveFileName(
            self, "Select propeller file", filter="Stereolithography file (*.stl)"
        )[0]
        if not path:
            return

        try:
            meshtoscale = mesh.Mesh(np.copy(self.stl_mesh.data))
            meshtoscale.vectors *= (
                1000 * 100 / 2.54
            )  # I think Tony's software is in like 10 thousandths of an inch?
            meshtoscale.save(path)
        except FileNotFoundError:
            QMessageBox.critical(self, "Error", "Failed to save stl file")
            return

    def set_mesh(self, blade_mesh):
        self.stl_file = None
        self.stl_mesh = blade_mesh

        if not self.popup and self.dialog:
            self.dialog.widget.set_mesh(blade_mesh)

        self.update_mesh_plot()

    def update_mesh_plot(self):
        # clear the view
        self.view.items = []

        stl_mesh = self.stl_mesh

        vertices = np.vstack((stl_mesh.v0, stl_mesh.v1, stl_mesh.v2))

        vertices = stl_mesh.vectors.reshape(-1, 3)
        faces = np.arange(len(vertices)).reshape(-1, 3)

        recalculated_normals = np.cross(
            stl_mesh.v1 - stl_mesh.v0, stl_mesh.v2 - stl_mesh.v0
        )
        recalculated_normals = recalculated_normals / (
            np.linalg.norm(recalculated_normals, axis=1)[:, None] + 1e-8
        )

        normals = np.repeat(recalculated_normals, 3, axis=0)

        # light direction to dot product with normals
        light_dir = np.array([1, 1, 1])
        light_dir = light_dir / np.linalg.norm(light_dir)

        min_intensity = np.inf
        max_intensity = -np.inf

        if self.arrowToggle.isChecked():
            line, arrow = create_arrow([0, 0, 0], [0, 0, np.max(stl_mesh.v1)])
            self.view.addItem(line)
            self.view.addItem(arrow)

        intensity = np.dot(normals, light_dir)
        intensity_per_face = intensity[::3]

        min_intensity = min(min_intensity, np.nanmin(intensity_per_face))
        max_intensity = max(max_intensity, np.nanmax(intensity_per_face))

        intensity_normalized = (intensity_per_face - min_intensity) / (
            max_intensity - min_intensity + 1e-8
        )

        # Map to colormap
        colors = cm.coolwarm(intensity_normalized)[:, :4]

        mesh_item = gl.GLMeshItem(
            vertexes=vertices,
            faces=faces,
            faceColors=colors,
            smooth=self.smoothToggle.isChecked(),
            drawEdges=self.edgeToggle.isChecked(),
            edgeColor=(1, 1, 1, 1),
        )
        self.view.addItem(mesh_item)
