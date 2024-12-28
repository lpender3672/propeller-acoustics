from PyQt6.QtWidgets import QApplication, QWidget, QGridLayout

from vis import STLViewerWidget
from input import PropInputTable, OpInputTable

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        layout = QGridLayout(self)
        
        self.prop_input = PropInputTable(self)
        self.op_input = OpInputTable(self)
        self.stl_viewer = STLViewerWidget()

        layout.addWidget(self.prop_input, 0, 0, 1, 1)
        layout.addWidget(self.op_input, 1, 0, 1, 1)
        layout.addWidget(self.stl_viewer, 0, 1, 2, 1)
        

        self.stl_viewer.set_stl_file("practical/designs/clarkY.stl")




def main():
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
