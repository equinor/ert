from PyQt4.QtCore import Qt
from PyQt4.QtGui import QMainWindow, QDockWidget, QTabWidget
from ert_gui.models.connectors.init import CaseSelectorModel
from ert_gui.tools.plot import PlotPanel, DataTypeKeysWidget, CaseSelectionWidget, PlotMetricsWidget, ExportPlotWidget, ExportPlot, CustomizePlotWidget
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
        """:type: list of PlotPanel"""

        self.__selected_plot_for_type = {}

        self.addPlotPanel("Ensemble plot", "gui/plots/simple_plot.html", short_name="Plot")
        self.addPlotPanel("Ensemble overview plot", "gui/plots/simple_overview_plot.html", short_name="oPlot")
        self.addPlotPanel("Histogram", "gui/plots/histogram.html", short_name="Histogram")
        self.addPlotPanel("Distribution", "gui/plots/gen_kw.html", short_name="Distribution")
        self.addPlotPanel("RFT plot", "gui/plots/rft.html", short_name="RFT")
        self.addPlotPanel("RFT overview plot", "gui/plots/rft_overview.html", short_name="oRFT")
        self.addPlotPanel("Ensemble plot", "gui/plots/gen_data.html", short_name="epGenData")
        self.addPlotPanel("Ensemble overview plot", "gui/plots/gen_data_overview.html", short_name="eopGenData")
        self.addPlotPanel("PCA plot", "gui/plots/pca.html", short_name="PCA")

        self.__data_type_keys_widget = DataTypeKeysWidget()
        self.__data_type_keys_widget.dataTypeKeySelected.connect(self.keySelected)
        self.addDock("Data types", self.__data_type_keys_widget)

        current_case = CaseSelectorModel().getCurrentChoice()
        self.__case_selection_widget = CaseSelectionWidget(current_case)
        self.__case_selection_widget.caseSelectionChanged.connect(self.caseSelectionChanged)
        plot_case_dock = self.addDock("Plot case", self.__case_selection_widget)

        self.__plot_metrics_widget = PlotMetricsWidget()
        self.__plot_metrics_widget.plotSettingsChanged.connect(self.plotSettingsChanged)
        plot_metrics_dock = self.addDock("Plot metrics", self.__plot_metrics_widget)

        self.__customize_plot_widget = CustomizePlotWidget()
        self.__customize_plot_widget.customPlotSettingsChanged.connect(self.plotSettingsChanged)
        customize_plot_dock = self.addDock("Customize", self.__customize_plot_widget)

        self.__export_plot_widget = ExportPlotWidget()
        self.__export_plot_widget.exportButtonPressed.connect(self.exportActivePlot)
        export_dock = self.addDock("Export Plot", self.__export_plot_widget)

        self.__exporter = None


        self.tabifyDockWidget(plot_case_dock, plot_metrics_dock)
        self.tabifyDockWidget(plot_case_dock, customize_plot_dock)
        self.tabifyDockWidget(plot_case_dock, export_dock)
        plot_case_dock.show()
        plot_case_dock.raise_()

        self.__plot_cases = self.__case_selection_widget.getPlotCaseNames()

    def plotSettingsChanged(self):
        plot_data_fetcher = PlotDataFetcher()
        data_key = self.__plot_metrics_widget.getDataKeyType()
        plot_data = plot_data_fetcher.getPlotDataForKeyAndCases(data_key, self.__plot_cases)
        plot_data.setParent(self)

        for plot_panel in self.__plot_panels:
            if plot_panel.isPlotVisible():
                model = plot_panel.getPlotBridge()
                model.setPlotData(plot_data)
                model.setCustomSettings(self.__customize_plot_widget.getCustomSettings())
                model.setPlotSettings(self.__plot_metrics_widget.getSettings())
                plot_panel.renderNow()

    def exportActivePlot(self):
        if self.__central_tab.currentIndex() > -1:
            active_plot =  self.__central_tab.currentWidget()
            assert isinstance(active_plot, PlotPanel)

            if self.__exporter is None:
                path = None
            else:
                path = self.__exporter.getCurrentPath()

            self.__exporter = ExportPlot(active_plot, self.__plot_metrics_widget.getSettings(), self.__customize_plot_widget.getCustomSettings(), path)

            self.__exporter.export()


    def addPlotPanel(self, name, path, short_name=None):
        if short_name is None:
            short_name = name

        plot_panel = PlotPanel(name, short_name, path)
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
            self.__customize_plot_widget.emitChange()


    def caseSelectionChanged(self):
        self.__plot_cases = self.__case_selection_widget.getPlotCaseNames()
        self.keySelected(self.__plot_metrics_widget.getDataKeyType())


    def showOrHidePlotTab(self, plot_panel, is_visible, show_plot):
        plot_panel.setPlotIsVisible(show_plot)
        if show_plot and not is_visible:
            index = self.__plot_panels.index(plot_panel)
            self.__central_tab.insertTab(index, plot_panel, plot_panel.getName())
        elif not show_plot and is_visible:
            index = self.__central_tab.indexOf(plot_panel)
            self.__central_tab.removeTab(index)


    def storePlotType(self, fetcher, key):
        if key is not None:
            if fetcher.isSummaryKey(key):
                self.__selected_plot_for_type["summary"] = self.__central_tab.currentWidget()
            elif fetcher.isBlockObservationKey(key):
                self.__selected_plot_for_type["block"] = self.__central_tab.currentWidget()
            elif fetcher.isGenKWKey(key):
                self.__selected_plot_for_type["gen_kw"] = self.__central_tab.currentWidget()
            elif fetcher.isGenDataKey(key):
                self.__selected_plot_for_type["gen_data"] = self.__central_tab.currentWidget()
            else:
                raise NotImplementedError("Key %s not supported." % key)

    def restorePlotType(self, fetcher, key):
        if key is not None:
            if fetcher.isSummaryKey(key):
                if "summary" in self.__selected_plot_for_type:
                    self.__central_tab.setCurrentWidget(self.__selected_plot_for_type["summary"])
                else:
                    if self.__central_tab.count() > 0:
                        self.__central_tab.setCurrentIndex(0)
            elif fetcher.isBlockObservationKey(key):
                if "block" in self.__selected_plot_for_type:
                    self.__central_tab.setCurrentWidget(self.__selected_plot_for_type["block"])
                else:
                    if self.__central_tab.count() > 0:
                        self.__central_tab.setCurrentIndex(0)
            elif fetcher.isGenKWKey(key):
                if "gen_kw" in self.__selected_plot_for_type:
                    self.__central_tab.setCurrentWidget(self.__selected_plot_for_type["gen_kw"])
                else:
                    if self.__central_tab.count() > 0:
                        self.__central_tab.setCurrentIndex(0)
            elif fetcher.isGenDataKey(key):
                if "gen_data" in self.__selected_plot_for_type:
                    self.__central_tab.setCurrentWidget(self.__selected_plot_for_type["gen_data"])
                else:
                    if self.__central_tab.count() > 0:
                        self.__central_tab.setCurrentIndex(0)
            else:
                raise NotImplementedError("Key %s not supported." % key)


    @may_take_a_long_time
    def keySelected(self, key):
        key = str(key)
        plot_data_fetcher = PlotDataFetcher()
        self.storePlotType(plot_data_fetcher, self.__plot_metrics_widget.getDataKeyType())

        show_pca = plot_data_fetcher.isPcaDataKey(key)

        for plot_panel in self.__plot_panels:
            visible = self.__central_tab.indexOf(plot_panel) > -1

            if plot_data_fetcher.isSummaryKey(key):
                show_plot = plot_panel.supportsPlotProperties(time=True, value=True, histogram=True, pca=show_pca)
                self.showOrHidePlotTab(plot_panel, visible, show_plot)

            elif plot_data_fetcher.isBlockObservationKey(key):
                show_plot = plot_panel.supportsPlotProperties(depth=True, value=True, pca=show_pca)
                self.showOrHidePlotTab(plot_panel, visible, show_plot)

            elif plot_data_fetcher.isGenKWKey(key):
                show_plot = plot_panel.supportsPlotProperties(value=True, histogram=True, pca=show_pca)
                self.showOrHidePlotTab(plot_panel, visible, show_plot)

            elif plot_data_fetcher.isGenDataKey(key):
                show_plot = plot_panel.supportsPlotProperties(index=True, pca=show_pca)
                self.showOrHidePlotTab(plot_panel, visible, show_plot)

            else:
                raise NotImplementedError("Key %s not supported." % key)

        self.restorePlotType(plot_data_fetcher, key)
        self.__plot_metrics_widget.setDataKeyType(key)

        if self.checkPlotStatus():
            self.plotSettingsChanged()

