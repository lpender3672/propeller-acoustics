from PyQt6.QtWidgets import QVBoxLayout, QWidget, QGridLayout, QCheckBox, QFileDialog, QPushButton, QMessageBox
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

        self.save_button = QPushButton("Save to .stl")
        self.save_button.clicked.connect(self.save_stl_file)


        #viewSettingLayout.addWidget(self.smoothToggle, 0, 0)
        viewSettingLayout.addWidget(self.edgeToggle, 0, 1)
        viewSettingLayout.addWidget(self.save_button, 0, 2)

        viewSettingWidget.setMaximumHeight(50)
        layout.addWidget(viewSettingWidget)
        layout.addWidget(self.view)

        self.view.setCameraPosition(distance=1)

        self.num_blades = 1

    def load_stl_file(self, stl_file):
        self.stl_file = stl_file
        self.stl_mesh = mesh.Mesh.from_file(stl_file)
        self.update_mesh_plot()
    
    def save_stl_file(self, stl_file):
        path = QFileDialog.getSaveFileName(self, "Select propeller file", filter="Stereolithography file (*.stl)")[0]
        if not path:
            return
        
        try:
            self.stl_mesh.save(path)
        except FileNotFoundError:
            QMessageBox.critical(self, "Error", "Failed to save stl file")
            return

    def set_mesh(self, blade_mesh, num_blades = 1):
        self.stl_file = None
        self.stl_mesh = blade_mesh
        self.num_blades = num_blades
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

        min_intensity = np.inf
        max_intensity = -np.inf

        intensities = np.zeros((self.num_blades, len(vertices) // 3))

        

        # first compute intensities to get colour range
        for i in range(self.num_blades):

            theta = 2 * np.pi * i / self.num_blades
            y_rot = np.array([
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)]
            ])
            
            rotated_normals = np.dot(normals, y_rot)
            intensity = np.dot(rotated_normals, light_dir)
            intensity_per_face = intensity[::3]

            min_intensity = min(min_intensity, np.nanmin(intensity_per_face))
            max_intensity = max(max_intensity, np.nanmax(intensity_per_face))

            intensities[i] = intensity_per_face

        #print(vertices.shape, faces.shape, colors.shape)
        for i in range(self.num_blades):

            theta = 2 * np.pi * i / self.num_blades
            y_rot = np.array([
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)]
            ])
            rotated_vertices = np.dot(vertices, y_rot)
        
            intensity_normalized = (intensities[i] - min_intensity) / (
            max_intensity - min_intensity + 1e-8
            )

            # Map to colormap
            colors = cm.coolwarm(intensity_normalized)[:, :4]

            mesh_item = gl.GLMeshItem(
                vertexes=rotated_vertices,
                faces=faces,
                faceColors=colors,
                smooth= self.smoothToggle.isChecked(),
                drawEdges= self.edgeToggle.isChecked(),
                edgeColor=(1, 1, 1, 1)
            )
            self.view.addItem(mesh_item)
        
