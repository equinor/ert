from qtpy.QtCore import Qt
from qtpy.QtWidgets import QMainWindow, QDockWidget, QTabWidget, QWidget, QVBoxLayout

from ert_gui.plottery.plots.ccsp import CrossCaseStatisticsPlot
from ert_gui.plottery.plots.distribution import DistributionPlot
from ert_gui.plottery.plots.ensemble import EnsemblePlot
from ert_gui.plottery.plots.gaussian_kde import GaussianKDEPlot
from ert_gui.plottery.plots.histogram import HistogramPlot
from ert_gui.plottery.plots.statistics import StatisticsPlot
from ert_shared import ERT
from ert_gui.ertwidgets import showWaitCursorWhileWaiting
from ert_gui.plottery import PlotContext, PlotConfig

from ert_gui.tools.plot import DataTypeKeysWidget, CaseSelectionWidget, PlotWidget
from ert_gui.tools.plot.customize import PlotCustomizer

from ert_gui.tools.plot.plot_api import PlotApi

CROSS_CASE_STATISTICS = "Cross Case Statistics"
DISTRIBUTION = "Distribution"
GAUSSIAN_KDE = "Gaussian KDE"
ENSEMBLE = "Ensemble"
HISTOGRAM = "Histogram"
STATISTICS = "Statistics"


class PlotWindow(QMainWindow):
    def __init__(self, config_file, parent):
        QMainWindow.__init__(self, parent)

        self._api = PlotApi(ERT.enkf_facade)

        self.setMinimumWidth(850)
        self.setMinimumHeight(650)

        self.setWindowTitle("Plotting - {}".format(config_file))
        self.activateWindow()
        self._key_definitions = self._api.all_data_type_keys()
        self._plot_customizer = PlotCustomizer(self, self._key_definitions)

        self._plot_customizer.settingsChanged.connect(self.keySelected)

        self._central_tab = QTabWidget()

        central_widget = QWidget()
        central_layout = QVBoxLayout()
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_widget.setLayout(central_layout)

        central_layout.addWidget(self._central_tab)

        self.setCentralWidget(central_widget)

        self._plot_widgets = []
        """:type: list of PlotWidget"""

        self.addPlotWidget(ENSEMBLE, EnsemblePlot())
        self.addPlotWidget(STATISTICS, StatisticsPlot())
        self.addPlotWidget(HISTOGRAM, HistogramPlot())
        self.addPlotWidget(GAUSSIAN_KDE, GaussianKDEPlot())
        self.addPlotWidget(DISTRIBUTION, DistributionPlot())
        self.addPlotWidget(CROSS_CASE_STATISTICS, CrossCaseStatisticsPlot())

        self._central_tab.currentChanged.connect(self.currentPlotChanged)

        cases = self._api.get_all_cases_not_running()
        case_names = [case["name"] for case in cases if not case["hidden"]]

        self._data_type_keys_widget = DataTypeKeysWidget(self._key_definitions)
        self._data_type_keys_widget.dataTypeKeySelected.connect(self.keySelected)
        self.addDock("Data types", self._data_type_keys_widget)
        self._case_selection_widget = CaseSelectionWidget(case_names)
        self._case_selection_widget.caseSelectionChanged.connect(self.keySelected)
        self.addDock("Plot case", self._case_selection_widget)

        current_plot_widget = self._plot_widgets[self._central_tab.currentIndex()]
        self._data_type_keys_widget.selectDefault()
        self._updateCustomizer(current_plot_widget)

    def currentPlotChanged(self):
        key_def = self.getSelectedKey()
        if key_def is None:
            return
        key = key_def["key"]

        for plot_widget in self._plot_widgets:
            index = self._central_tab.indexOf(plot_widget)

            if (
                index == self._central_tab.currentIndex()
                and plot_widget._plotter.dimensionality == key_def["dimensionality"]
            ):
                self._updateCustomizer(plot_widget)
                cases = self._case_selection_widget.getPlotCaseNames()
                case_to_data_map = {
                    case: self._api.data_for_key(case, key) for case in cases
                }
                if len(key_def["observations"]) > 0 and cases:
                    observations = self._api.observations_for_obs_keys(
                        cases[0], key_def["observations"]
                    )
                else:
                    observations = None

                plot_config = PlotConfig.createCopy(
                    self._plot_customizer.getPlotConfig()
                )
                plot_config.setTitle(key)
                plot_context = PlotContext(plot_config, cases, key)

                if key_def["has_refcase"]:
                    plot_context.refcase_data = self._api.refcase_data(key)

                case = plot_context.cases()[0] if plot_context.cases() else None
                plot_context.history_data = self._api.history_data(key, case)

                plot_context.log_scale = key_def["log_scale"]

                plot_widget.updatePlot(plot_context, case_to_data_map, observations)

    def _updateCustomizer(self, plot_widget):
        """@type plot_widget: PlotWidget"""
        key_def = self.getSelectedKey()
        if key_def is None:
            return
        index_type = key_def["index_type"]

        x_axis_type = PlotContext.UNKNOWN_AXIS
        y_axis_type = PlotContext.UNKNOWN_AXIS

        if plot_widget.name == ENSEMBLE:
            x_axis_type = index_type
            y_axis_type = PlotContext.VALUE_AXIS
        elif plot_widget.name == STATISTICS:
            x_axis_type = index_type
            y_axis_type = PlotContext.VALUE_AXIS
        elif plot_widget.name == DISTRIBUTION:
            y_axis_type = PlotContext.VALUE_AXIS
        elif plot_widget.name == CROSS_CASE_STATISTICS:
            y_axis_type = PlotContext.VALUE_AXIS
        elif plot_widget.name == HISTOGRAM:
            x_axis_type = PlotContext.VALUE_AXIS
            y_axis_type = PlotContext.COUNT_AXIS
        elif plot_widget.name == GAUSSIAN_KDE:
            x_axis_type = PlotContext.VALUE_AXIS
            y_axis_type = PlotContext.DENSITY_AXIS

        self._plot_customizer.setAxisTypes(x_axis_type, y_axis_type)

    def getSelectedKey(self):
        return self._data_type_keys_widget.getSelectedItem()

    def addPlotWidget(self, name, plotter, enabled=True):
        plot_widget = PlotWidget(name, plotter)
        plot_widget.customizationTriggered.connect(self.toggleCustomizeDialog)

        index = self._central_tab.addTab(plot_widget, name)
        self._plot_widgets.append(plot_widget)
        self._central_tab.setTabEnabled(index, enabled)

    def addDock(
        self,
        name,
        widget,
        area=Qt.LeftDockWidgetArea,
        allowed_areas=Qt.AllDockWidgetAreas,
    ):
        dock_widget = QDockWidget(name)
        dock_widget.setObjectName("%sDock" % name)
        dock_widget.setWidget(widget)
        dock_widget.setAllowedAreas(allowed_areas)
        dock_widget.setFeatures(
            QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetMovable
        )

        self.addDockWidget(area, dock_widget)
        return dock_widget

    @showWaitCursorWhileWaiting
    def keySelected(self):
        key_def = self.getSelectedKey()
        self._plot_customizer.switchPlotConfigHistory(key_def)

        available_widgets = [
            widget
            for widget in self._plot_widgets
            if widget._plotter.dimensionality == key_def["dimensionality"]
        ]
        self._central_tab.currentChanged.disconnect()
        for plot_widget in self._plot_widgets:
            self._central_tab.setTabEnabled(
                self._central_tab.indexOf(plot_widget), plot_widget in available_widgets
            )
        self._central_tab.currentChanged.connect(self.currentPlotChanged)
        self._central_tab.setCurrentWidget(available_widgets[0])

        self.currentPlotChanged()

    def toggleCustomizeDialog(self):
        self._plot_customizer.toggleCustomizationDialog()
