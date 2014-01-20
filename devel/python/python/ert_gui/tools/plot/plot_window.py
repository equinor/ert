from PyQt4.QtCore import Qt
from PyQt4.QtGui import QMainWindow, QDockWidget, QTabWidget
from ert_gui.models.connectors.init import CaseSelectorModel
from ert_gui.tools.plot import PlotPanel, DataTypeKeysWidget, CaseSelectionWidget, PlotMetricsWidget, ScaleTracker
from ert_gui.tools.plot.data import PlotDataFetcher
from ert_gui.widgets.util import may_take_a_long_time


class PlotWindow(QMainWindow):
    def __init__(self, parent):
        QMainWindow.__init__(self, parent)

        self.setMinimumWidth(750)
        self.setMinimumHeight(500)

        self.setWindowTitle("Plotting")
        self.activateWindow()

        self.__central_tab = QTabWidget()
        self.setCentralWidget(self.__central_tab)


        self.__plot_panels = []
        self.addPlotPanel("Ensemble plot", "gui/plots/simple_plot.html", short_name="Plot")
        self.addPlotPanel("Ensemble overview plot", "gui/plots/simple_overview_plot.html", short_name="oPlot")
        self.addPlotPanel("Histogram", "gui/plots/histogram.html", short_name="Histogram")

        self.__data_type_keys_widget = DataTypeKeysWidget()
        self.__data_type_keys_widget.dataTypeKeySelected.connect(self.keySelected)
        self.addDock("Data types", self.__data_type_keys_widget)

        current_case = CaseSelectorModel().getCurrentChoice()
        self.__case_selection_widget = CaseSelectionWidget(current_case)
        self.__case_selection_widget.caseSelectionChanged.connect(self.caseSelectionChanged)
        self.addDock("Plot case", self.__case_selection_widget)

        self.__plot_metrics_widget = PlotMetricsWidget()
        self.__plot_metrics_widget.plotScalesChanged.connect(self.scalesChanged)
        self.__plot_metrics_widget.reportStepTimeChanged.connect(self.reportStepTimeChanged)
        self.addDock("Plot metrics", self.__plot_metrics_widget)


        self.__data_type_key = None
        self.__plot_cases = self.__case_selection_widget.getPlotCaseNames()
        self.__value_scale_tracker = ScaleTracker("Value")
        self.__time_scale_tracker = ScaleTracker("Time")




    def addPlotPanel(self, name, path, short_name=None):
        if short_name is None:
            short_name = name

        plot_panel = PlotPanel(short_name, path)
        plot_panel.plotReady.connect(self.plotReady)
        self.__plot_panels.append(plot_panel)
        self.__central_tab.addTab(plot_panel, name)


    def addDock(self, name, widget, area=Qt.LeftDockWidgetArea, allowed_areas=Qt.AllDockWidgetAreas):
        dock_widget = QDockWidget(name)
        dock_widget.setObjectName("%sDock" % name)
        dock_widget.setWidget(widget)
        dock_widget.setAllowedAreas(allowed_areas)
        dock_widget.setFeatures(QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetMovable)

        self.addDockWidget(area, dock_widget)
        return dock_widget


    def checkPlotStatus(self):
        for plot_panel in self.__plot_panels:
            if not plot_panel.isReady():
                return False

        if len(self.__plot_cases) == 0:
            return False

        return True

    def plotReady(self):
        if self.checkPlotStatus():
            self.__data_type_keys_widget.selectDefault()


    def caseSelectionChanged(self):
        self.__plot_cases = self.__case_selection_widget.getPlotCaseNames()
        self.keySelected(self.__data_type_key)

    def scalesChanged(self):
        value_min = self.__plot_metrics_widget.getValueMin()
        value_max = self.__plot_metrics_widget.getValueMax()
        time_min = self.__plot_metrics_widget.getTimeMin()
        time_max = self.__plot_metrics_widget.getTimeMax()

        self.__value_scale_tracker.setScaleValues(self.__data_type_key, value_min, value_max)
        self.__time_scale_tracker.setScaleValues(self.__data_type_key, time_min, time_max)

        for plot_panel in self.__plot_panels:
            plot_panel.setValueScales(value_min, value_max)


    def reportStepTimeChanged(self):
        t = self.__plot_metrics_widget.getSelectedReportStepTime()

        for plot_panel in self.__plot_panels:
            plot_panel.setReportStepTime(t)


    @may_take_a_long_time
    def keySelected(self, key):
        self.__data_type_key = str(key)

        value_min = self.__value_scale_tracker.getMinimumScaleValue(self.__data_type_key)
        value_max = self.__value_scale_tracker.getMaximumScaleValue(self.__data_type_key)
        time_min = self.__time_scale_tracker.getMinimumScaleValue(self.__data_type_key)
        time_max = self.__time_scale_tracker.getMaximumScaleValue(self.__data_type_key)

        self.__plot_metrics_widget.updateScales(time_min, time_max, value_min, value_max)


        if self.checkPlotStatus():
            data = PlotDataFetcher().getPlotDataForKeyAndCases(self.__data_type_key, self.__plot_cases)
            data.setParent(self)

            for plot_panel in self.__plot_panels:
                plot_panel.setPlotData(data)

