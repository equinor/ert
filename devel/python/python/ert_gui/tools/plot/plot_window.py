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

        self.__central_tab = QTabWidget()
        self.__central_tab.currentChanged.connect(self.currentPlotChanged)

        central_widget = QWidget()
        central_layout = QVBoxLayout()
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_widget.setLayout(central_layout)

        central_layout.addWidget(self.__central_tab)

        self.setCentralWidget(central_widget)

        key_manager = ert.getKeyManager()
        """:type: ert.enkf.key_manager.KeyManager """

        self.__plot_widgets = []
        """:type: list of PlotWidget"""

        self.addPlotWidget("Ensemble", SummaryPlot.summaryEnsemblePlot, key_manager.isSummaryKey)
        self.addPlotWidget("Ensemble overview", SummaryPlot.summaryOverviewPlot, key_manager.isKeyWithObservations)
        self.addPlotWidget("Ensemble statistics", SummaryPlot.summaryStatisticsPlot, key_manager.isSummaryKey)


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
        self.__customize_plot_widget.customPlotSettingsChanged.connect(self.keySelected)
        customize_plot_dock = self.addDock("Customize", self.__customize_plot_widget)


        self.tabifyDockWidget(plot_case_dock, customize_plot_dock)

        plot_case_dock.show()
        plot_case_dock.raise_()

        self.__plot_widgets[self.__central_tab.currentIndex()].setActive()
        self.__data_type_keys_widget.selectDefault()



    def currentPlotChanged(self):
        for plot_widget in self.__plot_widgets:
            plot_widget.setActive(False)
            index = self.__central_tab.indexOf(plot_widget)

            if index == self.__central_tab.currentIndex():
                plot_widget.setActive()
                plot_widget.updatePlot()

    def createPlotContext(self, figure):
        key = self.getSelectedKey()
        cases = self.__case_selection_widget.getPlotCaseNames()
        plot_config = PlotConfig(key)
        self.applyCustomization(plot_config)
        return PlotContext(self.__ert, figure, plot_config, cases, key)

    def getSelectedKey(self):
        key = str(self.__data_type_keys_widget.getSelectedItem())
        return key

    def addPlotWidget(self, name, plotFunction, plotCondition, enabled=True):
        plot_widget = PlotWidget(name, plotFunction, plotCondition, self.createPlotContext)

        index = self.__central_tab.addTab(plot_widget, name)
        self.__plot_widgets.append(plot_widget)
        self.__central_tab.setTabEnabled(index, enabled)


    def addDock(self, name, widget, area=Qt.LeftDockWidgetArea, allowed_areas=Qt.AllDockWidgetAreas):
        dock_widget = QDockWidget(name)
        dock_widget.setObjectName("%sDock" % name)
        dock_widget.setWidget(widget)
        dock_widget.setAllowedAreas(allowed_areas)
        dock_widget.setFeatures(QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetMovable)

        self.addDockWidget(area, dock_widget)
        return dock_widget


    def applyCustomization(self, plot_config):
        custom = self.__customize_plot_widget.getCustomSettings()

        plot_config.setObservationsEnabled(custom["show_observations"])
        plot_config.setRefcaseEnabled(custom["show_refcase"])
        plot_config.setLegendEnabled(custom["show_legend"])
        plot_config.setGridEnabled(custom["show_grid"])


    @may_take_a_long_time
    def keySelected(self):
        key = self.getSelectedKey()

        for plot_widget in self.__plot_widgets:
            plot_widget.setDirty()
            index = self.__central_tab.indexOf(plot_widget)
            self.__central_tab.setTabEnabled(index, plot_widget.canPlotKey(key))

        for plot_widget in self.__plot_widgets:
            if plot_widget.canPlotKey(key):
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
