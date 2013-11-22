from PyQt4.QtCore import Qt
from PyQt4.QtGui import QMainWindow, QDockWidget
from ert_gui.models.connectors.plot import EnsembleSummaryPlot
from ert_gui.tools.plot import PlotPanel
from ert_gui.tools.plot import DataTypeKeysWidget
from ert_gui.widgets.util import may_take_a_long_time


class PlotWindow(QMainWindow):
    def __init__(self, parent):
        QMainWindow.__init__(self, parent)

        self.resize(500, 500)

        self.setWindowTitle("Plotting")
        self.activateWindow()

        self.plot_panel = PlotPanel("gui/plots/simple_plot.html")
        self.plot_panel.plotReady.connect(self.plotReady)
        self.setCentralWidget(self.plot_panel)

        self.data_type_keys_widget = DataTypeKeysWidget()
        self.data_type_keys_widget.dataTypeKeySelected.connect(self.keySelected)
        self.addDock("Data types", self.data_type_keys_widget)



    def addDock(self, name, widget, area=Qt.LeftDockWidgetArea, allowed_areas=Qt.AllDockWidgetAreas):
        dock_widget = QDockWidget(name)
        dock_widget.setObjectName("%sDock" % name)
        dock_widget.setWidget(widget)
        dock_widget.setAllowedAreas(allowed_areas)
        dock_widget.setFeatures(QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetMovable)

        self.addDockWidget(area, dock_widget)
        return dock_widget


    def plotReady(self):
        self.data_type_keys_widget.selectDefault()

    @may_take_a_long_time
    def keySelected(self, key):
        self.plot_panel.setPlotData(EnsembleSummaryPlot().getPlotDataForKey(str(key)))