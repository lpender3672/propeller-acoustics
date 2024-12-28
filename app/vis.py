from PyQt6.QtWidgets import QApplication, QVBoxLayout, QWidget, QGridLayout, QCheckBox
import pyqtgraph.opengl as gl
import numpy as np
from stl import mesh
from matplotlib import cm

class STLViewerWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.view = gl.GLViewWidget()

        viewSettingWidget = QWidget()
        viewSettingLayout = QGridLayout(viewSettingWidget)

        self.smoothToggle = QCheckBox("Smooth")
        self.smoothToggle.setChecked(False)
        self.smoothToggle.stateChanged.connect(self.update_mesh_plot)

        self.edgeToggle = QCheckBox("Draw Edges")
        self.edgeToggle.setChecked(False)
        self.edgeToggle.stateChanged.connect(self.update_mesh_plot)

        #viewSettingLayout.addWidget(self.smoothToggle, 0, 0)
        viewSettingLayout.addWidget(self.edgeToggle, 0, 1)

        viewSettingWidget.setMaximumHeight(30)
        layout.addWidget(viewSettingWidget)
        layout.addWidget(self.view)

        self.setMinimumHeight(400)
        self.setMinimumWidth(400)

        self.view.setCameraPosition(distance=0.1)

    def load_stl_file(self, stl_file):
        self.stl_file = stl_file
        self.stl_mesh = mesh.Mesh.from_file(stl_file)
        self.update_mesh_plot()

    def update_mesh_plot(self):
        # clear the view
        self.view.items = []

        stl_mesh = self.stl_mesh

        vertices = np.vstack((stl_mesh.v0, stl_mesh.v1, stl_mesh.v2))

        vertices = stl_mesh.vectors.reshape(-1, 3)
        faces = np.arange(len(vertices)).reshape(-1, 3)

        recalculated_normals = np.cross(stl_mesh.v1 - stl_mesh.v0, stl_mesh.v2 - stl_mesh.v0)
        recalculated_normals = recalculated_normals / np.linalg.norm(recalculated_normals, axis=1)[:, None]

        normals = np.repeat(recalculated_normals, 3, axis=0)

        # light direction to dot product with normals
        light_dir = np.array([1, 1, 1]) 
        light_dir = light_dir / np.linalg.norm(light_dir)

        intensity = np.dot(normals, light_dir)
        intensity_per_face = intensity[::3]
        colors = cm.coolwarm(intensity_per_face)[:, :4]

        #print(vertices.shape, faces.shape, colors.shape)

        mesh_item = gl.GLMeshItem(
            vertexes=vertices,
            faces=faces,
            faceColors=colors,
            smooth= self.smoothToggle.isChecked(),
            drawEdges= self.edgeToggle.isChecked(),
            edgeColor=(1, 1, 1, 1)
        )
        self.view.addItem(mesh_item)
        
