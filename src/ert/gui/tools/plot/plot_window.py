from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
from pandas import DataFrame
from PyQt6.QtCore import Qt
from PyQt6.QtCore import pyqtSlot as Slot
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QStyle,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ert.config.field import Field
from ert.dark_storage.common import get_storage_api_version
from ert.gui.ertwidgets import CopyButton, showWaitCursorWhileWaiting
from ert.services import ServerBootFail
from ert.utils import log_duration

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
    EverestGradientsPlot,
    GaussianKDEPlot,
    HistogramPlot,
    MisfitsPlot,
    StatisticsPlot,
    StdDevPlot,
    ValuesOverIterationsPlot,
)
from .widgets.everest_control_selection_widget import EverestControlSelectionWidget

CROSS_ENSEMBLE_STATISTICS = "Cross ensemble statistics"
DISTRIBUTION = "Distribution"
GAUSSIAN_KDE = "Gaussian KDE"
ENSEMBLE = "Ensemble"
HISTOGRAM = "Histogram"
STATISTICS = "Statistics"
STD_DEV = "Std Dev"
MISFITS = "Misfits"
EVEREST_RESPONSES_PLOT = "Batch responses"
EVEREST_CONTROLS_PLOT = "Batch controls"
EVEREST_GRADIENTS_PLOT = "Batch Gradients"

RESPONSE_DEFAULT = 0
GEN_KW_DEFAULT = 3
STD_DEV_DEFAULT = 7


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pathlib import Path

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
    qd.resize(450, 150)
    return qd


def open_error_dialog(title: str, content: str) -> None:
    qd = create_error_dialog(title, content)
    QApplication.restoreOverrideCursor()
    qd.exec()


def handle_exception(e: BaseException) -> None:
    if isinstance(e, TimeoutError):
        e.args = (
            "Plot API request timed out. Please check your connection ",
            "or the storage server status",
        )
    elif isinstance(e, ServerBootFail):
        e.args = ("The storage server failed to start",)
    logger.exception(e)  # noqa: LOG004
    open_error_dialog(type(e).__name__, str(e))


class PlotWindow(QMainWindow):
    @log_duration(logger, logging.INFO, "PlotWindow.__init__")
    def __init__(
        self, config_file: str, ens_path: Path, parent: QWidget | None
    ) -> None:
        super().__init__(parent)

        logger.info("PlotWindow __init__")
        self.setMinimumWidth(850)
        self.setMinimumHeight(650)
        self.setWindowTitle(f"Plotting - {config_file}")
        self.activateWindow()
        self._preferred_ensemble_x_axis_format = PlotContext.INDEX_AXIS
        self._api = PlotApi(ens_path)

        self.local_version = get_storage_api_version()

        if self._api.api_version != self.local_version:
            central_widget = QWidget()
            central_layout = QVBoxLayout()
            central_layout.setContentsMargins(20, 20, 20, 20)
            central_widget.setLayout(central_layout)
            label = QLabel(
                f"<b>Plot API version mismatch detected</b><br>"
                f"Runtime API version:<b>{self.local_version}</b><br>"
                f"Plot API version:<b>{self._api.api_version}</b><br><br>"
                "Unable to continue plotting operation"
            )
            label.setObjectName("plot_api_warning_label")
            icon_label = QLabel()

            style = QApplication.style()

            if style:
                warning_icon = style.standardIcon(
                    QStyle.StandardPixmap.SP_MessageBoxWarning
                )
                icon_label.setPixmap(warning_icon.pixmap(64, 64))

            central_layout.addWidget(icon_label)
            central_layout.addWidget(label)
            central_layout.addStretch(1)
            self.setCentralWidget(central_widget)
        else:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                self._key_definitions = (
                    self._api.responses_api_key_defs + self._api.parameters_api_key_defs
                )
            except BaseException as e:
                handle_exception(e)
                self._key_definitions = []
            QApplication.restoreOverrideCursor()

            is_everest = any(
                k.metadata.get("data_origin")
                in {"everest_parameters", "everest_objectives", "everest_constraints"}
                for k in self._key_definitions
            )

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

            if not is_everest:
                self.addPlotWidget(ENSEMBLE, EnsemblePlot())
                self.addPlotWidget(STATISTICS, StatisticsPlot())
                self.addPlotWidget(MISFITS, MisfitsPlot())
                self.addPlotWidget(HISTOGRAM, HistogramPlot())
                self.addPlotWidget(GAUSSIAN_KDE, GaussianKDEPlot())
                self.addPlotWidget(DISTRIBUTION, DistributionPlot())
                self.addPlotWidget(
                    CROSS_ENSEMBLE_STATISTICS, CrossEnsembleStatisticsPlot()
                )
                self.addPlotWidget(STD_DEV, StdDevPlot())
            else:
                self.addPlotWidget(ENSEMBLE, EnsemblePlot())
                self.addPlotWidget(EVEREST_CONTROLS_PLOT, ValuesOverIterationsPlot())
                self.addPlotWidget(EVEREST_RESPONSES_PLOT, ValuesOverIterationsPlot())
                self.addPlotWidget(EVEREST_GRADIENTS_PLOT, EverestGradientsPlot())

            self._central_tab.currentChanged.connect(self.currentTabChanged)
            self.logPlotTabUsage(self._central_tab.tabText(0), default=True)

            self._prev_tab_widget_index = -1
            self._current_tab_index = -1
            self._prev_key_dimensionality = -1
            self._prev_tab_widget_index_map: dict[int, int] = {}
            if is_everest:
                self._prev_tab_widget_index_map = {
                    1: 0,
                    2: 1,
                    3: 0,  # Fallback
                }
            else:
                self._prev_tab_widget_index_map = {
                    2: RESPONSE_DEFAULT,
                    1: GEN_KW_DEFAULT,
                    3: STD_DEV_DEFAULT,
                }

            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                ensembles = self._api.get_all_ensembles()
            except BaseException as e:
                handle_exception(e)
                ensembles = []
            QApplication.restoreOverrideCursor()

            plot_case_objects = [obj for obj in ensembles if not obj.hidden]

            self._data_type_keys_widget = DataTypeKeysWidget(self._key_definitions)
            self._data_type_keys_widget.dataTypeKeySelected.connect(self.keySelected)
            self.addDock("Data types", self._data_type_keys_widget)

            self._ensemble_selection_widget = EnsembleSelectionWidget(
                plot_case_objects,
                self._plot_customizer.getPlotConfig().getNumberOfColors(),
            )

            self._ensemble_selection_widget.ensembleSelectionChanged.connect(
                self.keySelected
            )
            self.addDock("Plot ensemble", self._ensemble_selection_widget)

            everest_parameters = [
                kd.parameter.name
                for kd in self._key_definitions
                if kd.parameter and kd.parameter.type == "everest_parameters"
            ]
            self._everest_control_selection_widget = EverestControlSelectionWidget(
                everest_parameters
            )
            self._everest_control_selection_widget.controlSelectionChanged.connect(
                self.updatePlot
            )
            self._everest_dock = self.addDock(
                "Everest Controls", self._everest_control_selection_widget
            )
            self._everest_dock.setVisible(False)
            self._data_type_keys_widget.selectDefault()

    def get_plot_api_version(self) -> str:
        return self._api.api_version

    @Slot(int)
    def currentTabChanged(self, index: int) -> None:
        self._current_tab_index = index
        self.updatePlot()
        self.logPlotTabUsage(self._central_tab.tabText(index))

    def logPlotTabUsage(self, tab_name: str, default: bool = False) -> None:
        msg = f"Plotwindow tab used: {tab_name}" + (" (default tab)" if default else "")
        logger.info(msg)

    @Slot(int)
    def layerIndexChanged(self, index: int | None) -> None:
        self.updatePlot(index)

    def updatePlot(self, layer: int | None = None) -> None:
        key_def = self.getSelectedKey()
        if key_def is None:
            return
        key = key_def.key

        plot_widget = cast(PlotWidget, self._central_tab.currentWidget())

        is_gradient_plot = plot_widget.name == EVEREST_GRADIENTS_PLOT
        self._everest_dock.setVisible(is_gradient_plot)

        if (
            plot_widget._plotter.dimensionality == key_def.dimensionality
            or (
                plot_widget.name
                in {
                    EVEREST_RESPONSES_PLOT,
                    EVEREST_CONTROLS_PLOT,
                    EVEREST_GRADIENTS_PLOT,
                }
            )
            or (key_def.metadata.get("data_origin") == "everest_batch_objectives")
        ):
            selected_ensembles = (
                self._ensemble_selection_widget.get_selected_ensembles()
            )
            ensemble_to_data_map: dict[EnsembleObject, pd.DataFrame] = {}

            if is_gradient_plot:
                selected_controls = (
                    self._everest_control_selection_widget.get_selected_controls()
                )
                plot_widget._plotter.set_selected_controls(selected_controls)  # type: ignore

            def fetch_data(
                ensemble: EnsembleObject,
            ) -> tuple[EnsembleObject, pd.DataFrame | BaseException | None]:
                try:
                    data = None
                    if is_gradient_plot:
                        data = self._api.data_for_gradient(ensemble.id, key)
                    elif (
                        key_def.response is not None
                        or key_def.metadata.get("data_origin")
                        == "everest_batch_objectives"
                    ):
                        data = self._api.data_for_response(
                            ensemble_id=ensemble.id,
                            response_key=key,
                            filter_on=key_def.filter_on,
                        )
                    elif key_def.parameter is not None and (
                        key_def.parameter.type
                        in {"gen_kw", "everest_parameters", "everest_objective"}
                    ):
                        data = self._api.data_for_parameter(
                            ensemble_id=ensemble.id,
                            parameter_key=key_def.parameter.name,
                        )
                except BaseException as e:
                    return ensemble, e

                return ensemble, data

            with ThreadPoolExecutor() as executor:
                for ensemble, result in executor.map(fetch_data, selected_ensembles):
                    if isinstance(result, BaseException):
                        handle_exception(result)
                    elif result is not None:
                        ensemble_to_data_map[ensemble] = result

            negative_values_in_data = False
            if key_def.parameter is not None and key_def.parameter.type == "gen_kw":
                for data in ensemble_to_data_map.values():
                    numeric = data.select_dtypes(include=["number"])
                    if not numeric.empty and numeric.le(0).any().any():
                        negative_values_in_data = True
                        break

            plot_widget._negative_values_in_data = negative_values_in_data
            observations = None
            if key_def.observations and selected_ensembles:
                try:
                    observations = self._api.observations_for_key(
                        [ensembles.id for ensembles in selected_ensembles], key
                    )
                except BaseException as e:
                    handle_exception(e)

            std_dev_images: dict[str, npt.NDArray[np.float32]] = {}

            if isinstance(key_def.parameter, Field):
                plot_widget.showLayerWidget.emit(True)
                layers = key_def.parameter.ertbox_params.nz
                plot_widget.updateLayerWidget.emit(layers)

                # select observations with locations
                obs_loc = self._api.observations_locations(
                    ensemble_ids=[selected_ensembles[0].id], param_cfg=key_def.parameter
                )

                if layer is None:
                    plot_widget.resetLayerWidget.emit()
                    layer = 0

                for ensemble in selected_ensembles:
                    try:
                        std_dev_images[ensemble.name] = self._api.std_dev_for_parameter(
                            key, ensemble.id, layer
                        )
                    except BaseException as e:
                        handle_exception(e)
            else:
                plot_widget.showLayerWidget.emit(False)

            plot_config = PlotConfig.createCopy(self._plot_customizer.getPlotConfig())
            plot_context = PlotContext(
                plot_config,
                selected_ensembles,
                self._ensemble_selection_widget.get_selected_ensembles_color_indexes(),
                key,
                layer,
            )

            # Check if key is a history key.
            # If it is it already has the data it needs
            if str(key).endswith("H") or "H:" in str(key):
                plot_context.history_data = DataFrame()
            else:
                try:
                    if self._api.has_history_data(key):
                        plot_context.history_data = self._api.history_data(
                            key,
                            [e.id for e in plot_context.ensembles()],
                        )

                except BaseException as e:
                    handle_exception(e)
                    plot_context.history_data = None

            if key_def.response is not None and key_def.response.type == "rft":
                plot_context.setXLabel(key.split(":")[-1])
                plot_context.setYLabel("TVD")
                plot_context.depth_y_axis = True
                for ekey, data in list(ensemble_to_data_map.items()):
                    ensemble_to_data_map[ekey] = data.interpolate(
                        method="linear", axis="columns"
                    )

            for data in ensemble_to_data_map.values():
                data = data.T

                if not data.empty and data.index.inferred_type == "datetime64":
                    self._preferred_ensemble_x_axis_format = PlotContext.DATE_AXIS
                    break

            self._updateCustomizer(plot_widget, self._preferred_ensemble_x_axis_format)

            plot_widget.updatePlot(
                plot_context,
                ensemble_to_data_map,
                observations,
                std_dev_images,
                key_def,
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
        | StdDevPlot
        | MisfitsPlot
        | ValuesOverIterationsPlot
        | EverestGradientsPlot,
        enabled: bool = True,
    ) -> None:
        plot_widget = PlotWidget(name, plotter)
        plot_widget.customizationTriggered.connect(self.toggleCustomizeDialog)
        plot_widget.layerIndexChanged.connect(self.layerIndexChanged)
        plot_widget.plotUpdateRequested.connect(self.updatePlot)

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
        dock_widget.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)

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
            and (key_def.observations or not widget._plotter.requires_observations)
        ]

        is_everest = key_def.metadata.get("data_origin") in {
            "everest_objectives",
            "everest_constraints",
        }
        everest_widget = next(
            (w for w in self._plot_widgets if w.name == EVEREST_RESPONSES_PLOT), None
        )

        if everest_widget:
            if (
                is_everest
                or key_def.metadata.get("data_origin") == "everest_batch_objectives"
            ):
                available_widgets = [everest_widget]
            elif everest_widget in available_widgets:
                available_widgets.remove(everest_widget)

        is_everest_control = (
            key_def.parameter is not None
            and key_def.parameter.type == "everest_parameters"
        )
        everest_control_widget = next(
            (w for w in self._plot_widgets if w.name == EVEREST_CONTROLS_PLOT), None
        )

        if everest_control_widget:
            if is_everest_control:
                available_widgets = [everest_control_widget]
            elif everest_control_widget in available_widgets:
                available_widgets.remove(everest_control_widget)

        everest_gradients_widget = next(
            (w for w in self._plot_widgets if w.name == EVEREST_GRADIENTS_PLOT), None
        )

        if everest_gradients_widget:
            if is_everest:
                if everest_gradients_widget not in available_widgets:
                    available_widgets.append(everest_gradients_widget)
            elif (
                key_def.metadata.get("data_origin") == "everest_batch_objectives"
                and everest_gradients_widget in available_widgets
            ) or everest_gradients_widget in available_widgets:
                available_widgets.remove(everest_gradients_widget)

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
        current_widget = self._central_tab.currentWidget()

        if 0 < self._prev_key_dimensionality != key_def.dimensionality:
            if self._current_tab_index == -1:
                self._current_tab_index = self._prev_tab_widget_index
            self._prev_tab_widget_index_map[self._prev_key_dimensionality] = (
                self._current_tab_index
            )
            current_widget = self._central_tab.widget(
                self._prev_tab_widget_index_map[key_def.dimensionality]
            )
            self._current_tab_index = -1

        if current_widget not in available_widgets and available_widgets:
            current_widget = available_widgets[0]

        self._central_tab.setCurrentWidget(current_widget)
        self._central_tab.currentChanged.connect(self.currentTabChanged)
        self._prev_tab_widget_index = self._central_tab.currentIndex()
        self._prev_key_dimensionality = key_def.dimensionality
        self.updatePlot()

    def toggleCustomizeDialog(self) -> None:
        self._plot_customizer.toggleCustomizationDialog()
