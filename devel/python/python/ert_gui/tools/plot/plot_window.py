from PyQt4.QtCore import Qt
from PyQt4.QtGui import QMainWindow, QDockWidget, QTabWidget, QWidget, QVBoxLayout

from ert_gui.models.connectors.init import CaseSelectorModel
from ert_gui.plottery import PlotContext, SummaryPlot, PlotConfig

from ert_gui.tools.plot import DataTypeKeysWidget, CaseSelectionWidget, CustomizePlotWidget, PlotWidget
from ert_gui.tools.plot import DataTypeKeysListModel
from ert_gui.widgets.util import may_take_a_long_time



class PlotWindow(QMainWindow):
    def __init__(self, ert, parent):
        QMainWindow.__init__(self, parent)

        self.__ert = ert
        """:type: ert.enkf.enkf_main.EnKFMain"""

        self.setMinimumWidth(750)
        self.setMinimumHeight(500)

        self.setWindowTitle("Plotting")
        self.activateWindow()

        # self.__plot_data = None
        #
        # self.__plot_metrics_tracker = PlotMetricsTracker()
        # self.__plot_metrics_tracker.addScaleType("value", float)
        # self.__plot_metrics_tracker.addScaleType("depth", float)
        # self.__plot_metrics_tracker.addScaleType("pca", float)
        # self.__plot_metrics_tracker.addScaleType("index", int)
        # self.__plot_metrics_tracker.addScaleType("count", int)
        # self.__plot_metrics_tracker.addScaleType("time", CTime)

        self.__central_tab = QTabWidget()
        # self.__central_tab.currentChanged.connect(self.currentPlotChanged)

        # self.__plot_panel_tracker = PlotPanelTracker(self.__central_tab)
        # self.__plot_panel_tracker.addKeyTypeTester("summary", PlotDataFetcher.isSummaryKey)
        # self.__plot_panel_tracker.addKeyTypeTester("block", PlotDataFetcher.isBlockObservationKey)
        # self.__plot_panel_tracker.addKeyTypeTester("gen_kw", PlotDataFetcher.isGenKWKey)
        # self.__plot_panel_tracker.addKeyTypeTester("gen_data", PlotDataFetcher.isGenDataKey)
        # self.__plot_panel_tracker.addKeyTypeTester("custom_pca", PlotDataFetcher.isCustomPcaDataKey)


        central_widget = QWidget()
        central_layout = QVBoxLayout()
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_widget.setLayout(central_layout)

        central_layout.addWidget(self.__central_tab)

        self.setCentralWidget(central_widget)


        self.__plot_widgets = {}
        """:type: dict of (int, PlotWidget)"""

        self.addPlotWidget("Ensemble plot", short_name="EP")
        self.addPlotWidget("Ensemble overview plot", short_name="EOP", enabled=False)
        # self.addPlotPanel("Ensemble statistics", "gui/plots/ensemble_statistics_plot.html", short_name="ES")
        # self.addPlotPanel("Histogram", "gui/plots/histogram.html", short_name="Histogram")
        # self.addPlotPanel("Distribution", "gui/plots/gen_kw.html", short_name="Distribution")
        # self.addPlotPanel("RFT plot", "gui/plots/rft.html", short_name="RFT")
        # self.addPlotPanel("RFT overview plot", "gui/plots/rft_overview.html", short_name="oRFT")
        # self.addPlotPanel("Ensemble plot", "gui/plots/gen_data.html", short_name="epGenData")
        # self.addPlotPanel("Ensemble overview plot", "gui/plots/gen_data_overview.html", short_name="eopGenData")
        # self.addPlotPanel("Ensemble statistics", "gui/plots/gen_data_statistics_plot.html", short_name="esGenData")
        # self.addPlotPanel("PCA plot", "gui/plots/pca.html", short_name="PCA")

        self.__data_types_key_model = DataTypeKeysListModel(ert)

        self.__data_type_keys_widget = DataTypeKeysWidget(self.__data_types_key_model)
        self.__data_type_keys_widget.dataTypeKeySelected.connect(self.keySelected)
        self.addDock("Data types", self.__data_type_keys_widget)

        current_case = CaseSelectorModel().getCurrentChoice()
        self.__case_selection_widget = CaseSelectionWidget(current_case)
        self.__case_selection_widget.caseSelectionChanged.connect(self.keySelected)
        plot_case_dock = self.addDock("Plot case", self.__case_selection_widget)

        self.__customize_plot_widget = CustomizePlotWidget()
        # self.__customize_plot_widget.customPlotSettingsChanged.connect(self.plotSettingsChanged)
        customize_plot_dock = self.addDock("Customize", self.__customize_plot_widget)

        # self.__toolbar.exportClicked.connect(self.exportActivePlot)
        # self.__toolbar.plotScalesChanged.connect(self.plotSettingsChanged)
        # self.__toolbar.reportStepChanged.connect(self.plotSettingsChanged)

        # self.__exporter = None
        self.tabifyDockWidget(plot_case_dock, customize_plot_dock)

        plot_case_dock.show()
        plot_case_dock.raise_()

        self.__data_type_keys_widget.selectDefault()

        # self.__plot_cases = self.__case_selection_widget.getPlotCaseNames()


    # def fetchDefaultXScales(self):
    #     if self.__plot_data is not None:
    #         active_plot = self.getActivePlot()
    #         x_axis_type_name = active_plot.xAxisType()
    #
    #         x_axis_type = self.__plot_metrics_tracker.getType(x_axis_type_name)
    #
    #         if x_axis_type is not None:
    #             x_min, x_max = active_plot.getXScales(self.__plot_data)
    #
    #             if x_axis_type is CTime and x_min is not None and x_max is not None:
    #                 x_min = int(x_min)
    #                 x_max = int(x_max)
    #
    #             self.__plot_metrics_tracker.setScalesForType(x_axis_type_name, x_min, x_max)
    #


    # def fetchDefaultYScales(self):
    #         if self.__plot_data is not None:
    #             active_plot = self.getActivePlot()
    #             y_axis_type_name = active_plot.yAxisType()
    #
    #             y_axis_type = self.__plot_metrics_tracker.getType(y_axis_type_name)
    #
    #             if y_axis_type is not None:
    #                 y_min, y_max = active_plot.getYScales(self.__plot_data)
    #
    #                 if y_axis_type is CTime and y_min is not None and y_max is not None:
    #                     y_min = int(y_min)
    #                     y_max = int(y_max)
    #
    #                 self.__plot_metrics_tracker.setScalesForType(y_axis_type_name, y_min, y_max)
    #
    #
    # def resetCurrentScales(self):
    #     active_plot = self.getActivePlot()
    #     x_axis_type_name = active_plot.xAxisType()
    #     y_axis_type_name = active_plot.yAxisType()
    #
    #     self.__plot_metrics_tracker.resetScaleType(x_axis_type_name)
    #     self.__plot_metrics_tracker.resetScaleType(y_axis_type_name)
    #
    #     self.currentPlotChanged()
    #     self.plotSettingsChanged()
    #
    #
    # def getActivePlot(self):
    #     """ @rtype: PlotPanel """
    #     if not self.__central_tab.currentIndex() > -1:
    #         raise AssertionError("No plot selected!")
    #
    #     active_plot =  self.__central_tab.currentWidget()
    #     assert isinstance(active_plot, PlotPanel)
    #
    #     return active_plot
    #
    #
    # def currentPlotChanged(self):
    #     active_plot = self.getActivePlot()
    #
    #     if active_plot.isReady():
    #         x_axis_type_name = active_plot.xAxisType()
    #         x_axis_type = self.__plot_metrics_tracker.getType(x_axis_type_name)
    #
    #         y_axis_type_name = active_plot.yAxisType()
    #         y_axis_type = self.__plot_metrics_tracker.getType(y_axis_type_name)
    #
    #         if not self.__plot_metrics_tracker.hasScale(x_axis_type_name):
    #             self.fetchDefaultXScales()
    #
    #         if not self.__plot_metrics_tracker.hasScale(y_axis_type_name):
    #             self.fetchDefaultYScales()
    #
    #         x_min, x_max = self.__plot_metrics_tracker.getScalesForType(x_axis_type_name)
    #         y_min, y_max = self.__plot_metrics_tracker.getScalesForType(y_axis_type_name)
    #
    #         self.__toolbar.setToolBarOptions(x_axis_type, y_axis_type, active_plot.isReportStepCapable() and self.__plot_metrics_tracker.dataTypeSupportsReportStep())
    #         self.__toolbar.setScales(x_min, x_max, y_min, y_max)


    # def plotSettingsChanged(self):
    #     x_min, x_max = self.__toolbar.getXScales()
    #     y_min, y_max = self.__toolbar.getYScales()
    #
    #     active_plot = self.getActivePlot()
    #
    #     x_axis_type_name = active_plot.xAxisType()
    #     y_axis_type_name = active_plot.yAxisType()
    #
    #     self.__plot_metrics_tracker.setScalesForType(x_axis_type_name, x_min, x_max)
    #     self.__plot_metrics_tracker.setScalesForType(y_axis_type_name, y_min, y_max)
    #
    #     self.updatePlots()
    #
    #
    # def updatePlots(self):
    #     report_step = self.__toolbar.getReportStep()
    #
    #     for plot_panel in self.__plot_panels:
    #         if plot_panel.isPlotVisible():
    #             model = plot_panel.getPlotBridge()
    #             model.setPlotData(self.__plot_data)
    #             model.setCustomSettings(self.__customize_plot_widget.getCustomSettings())
    #             model.setReportStepTime(report_step)
    #
    #             x_axis_type_name = plot_panel.xAxisType()
    #             y_axis_type_name = plot_panel.yAxisType()
    #
    #             x_min, x_max = self.__plot_metrics_tracker.getScalesForType(x_axis_type_name)
    #             y_min, y_max = self.__plot_metrics_tracker.getScalesForType(y_axis_type_name)
    #
    #             model.setScales(x_min, x_max, y_min, y_max)
    #
    #             plot_panel.renderNow()
    #



    def addPlotWidget(self, name, short_name=None, enabled=True):
        if short_name is None:
            short_name = name

        plot_widget = PlotWidget(name, short_name)

        index = self.__central_tab.addTab(plot_widget, name)
        self.__plot_widgets[index] = plot_widget
        self.__central_tab.setTabEnabled(index, enabled)


    def addDock(self, name, widget, area=Qt.LeftDockWidgetArea, allowed_areas=Qt.AllDockWidgetAreas):
        dock_widget = QDockWidget(name)
        dock_widget.setObjectName("%sDock" % name)
        dock_widget.setWidget(widget)
        dock_widget.setAllowedAreas(allowed_areas)
        dock_widget.setFeatures(QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetMovable)

        self.addDockWidget(area, dock_widget)
        return dock_widget

    def __newFigure(self):
        """ :rtype: matplotlib.figure.Figure"""
        plot_widget = self.__plot_widgets[0]
        plot_widget.resetPlot()
        return plot_widget.getFigure()



    @may_take_a_long_time
    def keySelected(self, key=None):
        if key is None:
            key = self.__data_type_keys_widget.getSelectedItem()
        key = str(key)

        print("Plotting key: %s" % key)

        cases = self.__case_selection_widget.getPlotCaseNames()

        if self.__data_types_key_model.isSummaryKey(key):
            plot_widget = self.__plot_widgets[0]
            figure = self.__newFigure()
            plot_config = PlotConfig(key)
            plot_ctx = PlotContext(self.__ert, figure, plot_config, cases, key)
            SummaryPlot.summaryEnsemblePlot(plot_ctx)
            plot_widget.updatePlot()


        # old_data_type_key = self.__plot_metrics_tracker.getDataTypeKey()
        # self.__plot_metrics_tracker.setDataTypeKey(key)
        #
        # plot_data_fetcher = PlotDataFetcher()
        # self.__plot_data = plot_data_fetcher.getPlotDataForKeyAndCases(key, self.__plot_cases)
        # self.__plot_data.setParent(self)
        #
        # self.__central_tab.blockSignals(True)
        #
        # self.__plot_panel_tracker.storePlotType(plot_data_fetcher, old_data_type_key)
        #
        # for plot_panel in self.__plot_panels:
        #     self.showOrHidePlotTab(plot_panel, False, True)
        #
        # self.__plot_metrics_tracker.setDataTypeKeySupportsReportSteps(plot_data_fetcher.dataTypeKeySupportsReportSteps(key))
        # show_pca = plot_data_fetcher.isPcaDataKey(key)
        # for plot_panel in self.__plot_panels:
        #     visible = self.__central_tab.indexOf(plot_panel) > -1
        #
        #     if plot_data_fetcher.isSummaryKey(key):
        #         show_plot = plot_panel.supportsPlotProperties(time=True, value=True, histogram=True, pca=show_pca)
        #         self.showOrHidePlotTab(plot_panel, visible, show_plot)
        #
        #     elif plot_data_fetcher.isBlockObservationKey(key):
        #         show_plot = plot_panel.supportsPlotProperties(depth=True, value=True, pca=show_pca)
        #         self.showOrHidePlotTab(plot_panel, visible, show_plot)
        #
        #     elif plot_data_fetcher.isGenKWKey(key):
        #         show_plot = plot_panel.supportsPlotProperties(value=True, histogram=True, pca=show_pca)
        #         self.showOrHidePlotTab(plot_panel, visible, show_plot)
        #
        #     elif plot_data_fetcher.isGenDataKey(key):
        #         show_plot = plot_panel.supportsPlotProperties(index=True, pca=show_pca)
        #         self.showOrHidePlotTab(plot_panel, visible, show_plot)
        #
        #     elif plot_data_fetcher.isPcaDataKey(key):
        #         show_plot = plot_panel.supportsPlotProperties(pca=show_pca)
        #         self.showOrHidePlotTab(plot_panel, visible, show_plot)
        #
        #     else:
        #         raise NotImplementedError("Key %s not supported." % key)
        #
        # self.__plot_panel_tracker.restorePlotType(plot_data_fetcher, key)
        #
        # self.__central_tab.blockSignals(False)
        # self.currentPlotChanged()
        #
        # if self.checkPlotStatus():
        #     self.plotSettingsChanged()
