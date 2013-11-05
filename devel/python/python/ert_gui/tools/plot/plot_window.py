from PyQt4.QtGui import QMainWindow
from ert_gui.tools.plot import PlotPanel


class PlotWindow(QMainWindow):
    def __init__(self, parent):
        QMainWindow.__init__(self, parent)

        self.resize(500, 500)
        self.plot_panel = PlotPanel()
        self.setCentralWidget(self.plot_panel)
        self.setWindowTitle("Plotting")
        self.activateWindow()

