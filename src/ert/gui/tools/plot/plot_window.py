import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from httpx import RequestError
from pandas import DataFrame
from qtpy.QtCore import Qt, Slot
from qtpy.QtWidgets import QDockWidget, QMainWindow, QTabWidget, QWidget

from ert.gui.ertwidgets import showWaitCursorWhileWaiting

from .customize import PlotCustomizer
from .data_type_keys_widget import DataTypeKeysWidget
from .plot_api import EnsembleObject, PlotApi, PlotApiKeyDefinition
from .plot_ensemble_selection_widget import EnsembleSelectionWidget
from .plot_widget import PlotWidget
from .plottery import PlotConfig, PlotContext
from .plottery.plots import (
    CrossEnsembleStatisticsPlot,
    DistributionPlot,
    EnsemblePlot,
    GaussianKDEPlot,
    HistogramPlot,
    StatisticsPlot,
    StdDevPlot,
)

CROSS_ENSEMBLE_STATISTICS = "Cross ensemble statistics"
DISTRIBUTION = "Distribution"
GAUSSIAN_KDE = "Gaussian KDE"
ENSEMBLE = "Ensemble"
HISTOGRAM = "Histogram"
STATISTICS = "Statistics"
STD_DEV = "Std Dev"

logger = logging.getLogger(__name__)

from qtpy.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QTextEdit,
    QVBoxLayout,
)

from ert.gui.ertwidgets import CopyButton

if TYPE_CHECKING:
    import numpy.typing as npt


class _CopyButton(CopyButton):
    def __init__(self, text_edit: QTextEdit) -> None:
        super().__init__()
        self.text_edit = text_edit

    def copy(self) -> None:
        self.copy_text(self.text_edit.toPlainText())


def create_error_dialog(title: str, content: str) -> QDialog:
    qd = QDialog()
    qd.setModal(True)
    qd.setSizeGripEnabled(True)
    layout = QVBoxLayout()
    top_layout = QHBoxLayout()
    top_layout.addWidget(QLabel(title))

    text = QTextEdit()
    text.setText(content)
    text.setReadOnly(True)

    copy_button = _CopyButton(text)
    copy_button.setObjectName("copy_button")
    top_layout.addWidget(copy_button)
    top_layout.addStretch(-1)

    layout.addLayout(top_layout)
    layout.addWidget(text)

    qd.setLayout(layout)
    return qd


def open_error_dialog(title: str, content: str) -> None:
    qd = create_error_dialog(title, content)
    QApplication.restoreOverrideCursor()
    qd.exec()


class PlotWindow(QMainWindow):
    def __init__(self, config_file: str, parent: QWidget | None):
        QMainWindow.__init__(self, parent)
        t = time.perf_counter()

        logger.info("PlotWindow __init__")
        self.setMinimumWidth(850)
        self.setMinimumHeight(650)
        self.setWindowTitle(f"Plotting - {config_file}")
        self.activateWindow()
        self._preferred_ensemble_x_axis_format = PlotContext.INDEX_AXIS
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self._api = PlotApi()
            self._key_definitions = self._api.all_data_type_keys()
        except (RequestError, TimeoutError) as e:
            logger.exception(f"plot api request failed: {e}")
            open_error_dialog("Request failed", str(e))
            self._key_definitions = []
        QApplication.restoreOverrideCursor()

        self._plot_customizer = PlotCustomizer(self, self._key_definitions)

        self._plot_customizer.settingsChanged.connect(self.keySelected)

        self._central_tab = QTabWidget()

        central_widget = QWidget()
        central_layout = QVBoxLayout()
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_widget.setLayout(central_layout)

        central_layout.addWidget(self._central_tab)

        self.setCentralWidget(central_widget)

        self._plot_widgets: list[PlotWidget] = []

        self.addPlotWidget(ENSEMBLE, EnsemblePlot())
        self.addPlotWidget(STATISTICS, StatisticsPlot())
        self.addPlotWidget(HISTOGRAM, HistogramPlot())
        self.addPlotWidget(GAUSSIAN_KDE, GaussianKDEPlot())
        self.addPlotWidget(DISTRIBUTION, DistributionPlot())
        self.addPlotWidget(CROSS_ENSEMBLE_STATISTICS, CrossEnsembleStatisticsPlot())
        self.addPlotWidget(STD_DEV, StdDevPlot())
        self._central_tab.currentChanged.connect(self.currentTabChanged)
        self._prev_tab_widget: QWidget | None = None

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            ensembles = self._api.get_all_ensembles()
        except (RequestError, TimeoutError) as e:
            logger.exception(e)
            open_error_dialog("Request failed", str(e))
            ensembles = []
        QApplication.restoreOverrideCursor()

        plot_case_objects = [obj for obj in ensembles if not obj.hidden]

        self._data_type_keys_widget = DataTypeKeysWidget(self._key_definitions)
        self._data_type_keys_widget.dataTypeKeySelected.connect(self.keySelected)
        self.addDock("Data types", self._data_type_keys_widget)
        self._ensemble_selection_widget = EnsembleSelectionWidget(plot_case_objects)

        self._ensemble_selection_widget.ensembleSelectionChanged.connect(
            self.keySelected
        )
        self.addDock("Plot ensemble", self._ensemble_selection_widget)

        self._data_type_keys_widget.selectDefault()

        logger.info(f"PlotWindow __init__ done. time={time.perf_counter() -t}")

    @Slot(int)
    def currentTabChanged(self, index: Any) -> None:
        self.updatePlot()

    @Slot(int)
    def layerIndexChanged(self, index: int | None) -> None:
        self.updatePlot(index)

    def updatePlot(self, layer: int | None = None) -> None:
        key_def = self.getSelectedKey()
        if key_def is None:
            return
        key = key_def.key

        plot_widget = self._central_tab.currentWidget()
        assert plot_widget is not None

        if plot_widget._plotter.dimensionality == key_def.dimensionality:
            selected_ensembles = (
                self._ensemble_selection_widget.get_selected_ensembles()
            )
            ensemble_to_data_map: dict[EnsembleObject, pd.DataFrame] = {}
            for ensemble in selected_ensembles:
                try:
                    ensemble_to_data_map[ensemble] = self._api.data_for_key(
                        ensemble.id, key
                    )
                except (RequestError, TimeoutError) as e:
                    logger.exception(f"plot api request failed: {e}")
                    open_error_dialog("Request failed", f"{e}")

            observations = None
            if key_def.observations and selected_ensembles:
                try:
                    observations = self._api.observations_for_key(
                        [ensembles.id for ensembles in selected_ensembles], key
                    )
                except (RequestError, TimeoutError) as e:
                    logger.exception(f"plot api request failed: {e}")
                    open_error_dialog("Request failed", f"{e}")

            std_dev_images: dict[str, npt.NDArray[np.float32]] = {}
            if "FIELD" in key_def.metadata["data_origin"]:
                plot_widget.showLayerWidget.emit(True)

                layers = key_def.metadata["nz"]
                plot_widget.updateLayerWidget.emit(layers)

                if layer is None:
                    plot_widget.resetLayerWidget.emit()
                    layer = 0

                for ensemble in selected_ensembles:
                    try:
                        std_dev_images[ensemble.name] = self._api.std_dev_for_parameter(
                            key, ensemble.id, layer
                        )
                    except (RequestError, TimeoutError) as e:
                        logger.exception(f"plot api request failed: {e}")
                        open_error_dialog("Request failed", f"{e}")
            else:
                plot_widget.showLayerWidget.emit(False)

            plot_config = PlotConfig.createCopy(self._plot_customizer.getPlotConfig())
            plot_context = PlotContext(plot_config, selected_ensembles, key, layer)

            # Check if key is a history key.
            # If it is it already has the data it needs
            if str(key).endswith("H") or "H:" in str(key):
                plot_context.history_data = DataFrame()
            else:
                try:
                    plot_context.history_data = self._api.history_data(
                        key,
                        [e.id for e in plot_context.ensembles()],
                    )

                except (RequestError, TimeoutError) as e:
                    logger.exception(f"plot api request failed: {e}")
                    open_error_dialog("Request failed", f"{e}")
                    plot_context.history_data = None

            plot_context.log_scale = key_def.log_scale

            for data in ensemble_to_data_map.values():
                data = data.T

                if not data.empty and data.index.inferred_type == "datetime64":
                    self._preferred_ensemble_x_axis_format = PlotContext.DATE_AXIS
                    break

            self._updateCustomizer(plot_widget, self._preferred_ensemble_x_axis_format)

            plot_widget.updatePlot(
                plot_context, ensemble_to_data_map, observations, std_dev_images
            )

    def _updateCustomizer(
        self, plot_widget: PlotWidget, preferred_x_axis_format: str
    ) -> None:
        x_axis_type = PlotContext.UNKNOWN_AXIS
        y_axis_type = PlotContext.UNKNOWN_AXIS

        if plot_widget.name in {ENSEMBLE, STATISTICS}:
            x_axis_type = preferred_x_axis_format
            y_axis_type = PlotContext.VALUE_AXIS
        elif plot_widget.name in {DISTRIBUTION, CROSS_ENSEMBLE_STATISTICS}:
            y_axis_type = PlotContext.VALUE_AXIS
        elif plot_widget.name == HISTOGRAM:
            x_axis_type = PlotContext.VALUE_AXIS
            y_axis_type = PlotContext.COUNT_AXIS
        elif plot_widget.name == GAUSSIAN_KDE:
            x_axis_type = PlotContext.VALUE_AXIS
            y_axis_type = PlotContext.DENSITY_AXIS

        self._plot_customizer.setAxisTypes(x_axis_type, y_axis_type)

    def getSelectedKey(self) -> PlotApiKeyDefinition | None:
        return self._data_type_keys_widget.getSelectedItem()

    def addPlotWidget(
        self,
        name: str,
        plotter: EnsemblePlot
        | StatisticsPlot
        | HistogramPlot
        | GaussianKDEPlot
        | DistributionPlot
        | CrossEnsembleStatisticsPlot
        | StdDevPlot,
        enabled: bool = True,
    ) -> None:
        plot_widget = PlotWidget(name, plotter)
        plot_widget.customizationTriggered.connect(self.toggleCustomizeDialog)
        plot_widget.layerIndexChanged.connect(self.layerIndexChanged)

        index = self._central_tab.addTab(plot_widget, name)
        self._plot_widgets.append(plot_widget)
        self._central_tab.setTabEnabled(index, enabled)

    def addDock(
        self,
        name: str,
        widget: QWidget,
        area: Qt.DockWidgetArea = Qt.DockWidgetArea.LeftDockWidgetArea,
        allowed_areas: Qt.DockWidgetArea = Qt.DockWidgetArea.AllDockWidgetAreas,
    ) -> QDockWidget:
        dock_widget = QDockWidget(name)
        dock_widget.setObjectName(f"{name}Dock")
        dock_widget.setWidget(widget)
        dock_widget.setAllowedAreas(allowed_areas)
        dock_widget.setFeatures(
            QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetMovable
        )

        self.addDockWidget(area, dock_widget)
        return dock_widget

    @showWaitCursorWhileWaiting
    def keySelected(self) -> None:
        key_def = self.getSelectedKey()
        if key_def is None:
            return
        self._plot_customizer.switchPlotConfigHistory(key_def)

        available_widgets = [
            widget
            for widget in self._plot_widgets
            if widget._plotter.dimensionality == key_def.dimensionality
        ]

        current_widget = self._central_tab.currentWidget()

        # Enabling/disabling tab triggers the
        # currentTabChanged event which also triggers
        # the updatePlot, which is slow and redundant.
        # Therefore, we disable this signal because this
        # part is only supposed to set which tabs are
        # enabled according to the available widgets.
        self._central_tab.currentChanged.disconnect()
        for plot_widget in self._plot_widgets:
            self._central_tab.setTabEnabled(
                self._central_tab.indexOf(plot_widget), plot_widget in available_widgets
            )
        self._central_tab.currentChanged.connect(self.currentTabChanged)

        # Remember which tab widget was selected when switching between
        # both same and different data-types.
        if current_widget in available_widgets:
            self._central_tab.setCurrentWidget(current_widget)
        else:
            if self._prev_tab_widget is None:
                self._central_tab.setCurrentWidget(available_widgets[0])
            else:
                self._central_tab.setCurrentWidget(self._prev_tab_widget)
            self._prev_tab_widget = current_widget

        self.updatePlot()

    def toggleCustomizeDialog(self) -> None:
        self._plot_customizer.toggleCustomizationDialog()
