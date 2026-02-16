import logging
import sys
import traceback
from typing import TYPE_CHECKING, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.backends.backend_qt5agg import (  # type: ignore
    FigureCanvas,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure
from PyQt6.QtCore import QStringListModel, Qt, pyqtBoundSignal
from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtCore import pyqtSlot as Slot
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)
from typing_extensions import override

from ert.gui.tools.plot.plottery.plots import EverestGradientsPlot

from .plot_api import EnsembleObject, PlotApiKeyDefinition

if TYPE_CHECKING:
    from .plottery import PlotContext
    from .plottery.plots.cesp import CrossEnsembleStatisticsPlot
    from .plottery.plots.distribution import DistributionPlot
    from .plottery.plots.ensemble import EnsemblePlot
    from .plottery.plots.gaussian_kde import GaussianKDEPlot
    from .plottery.plots.histogram import HistogramPlot
    from .plottery.plots.misfits import MisfitsPlot
    from .plottery.plots.statistics import StatisticsPlot
    from .plottery.plots.std_dev import StdDevPlot
    from .plottery.plots.values_over_iteration_plot import (
        ValuesOverIterationsPlot,
    )

logger = logging.getLogger(__name__)


class CustomNavigationToolbar(NavigationToolbar2QT):
    customizationTriggered = Signal()
    layerIndexChanged = Signal(int)

    def __init__(
        self,
        canvas: FigureCanvas,
        parent: QWidget | None,
        coordinates: bool = True,
    ) -> None:
        super().__init__(canvas, parent, coordinates)  # type: ignore

        gear = QIcon("img:edit.svg")
        customize_action = QAction(gear, "Customize", self)
        customize_action.setToolTip("Customize plot settings")
        customize_action.triggered.connect(self.customizationTriggered)
        customize_action.triggered.connect(
            lambda: self.logToolbarUsage(customize_action.text())
        )

        layer_combobox = QComboBox()
        self._model = QStringListModel()
        layer_combobox.setModel(self._model)
        layer_combobox.currentIndexChanged.connect(self.layerIndexChanged)

        for action in self.actions():
            if str(action.text()).lower() == "subplots":
                self.removeAction(action)

            if str(action.text()).lower() == "customize":
                self.insertAction(action, customize_action)
                self.removeAction(action)

            # insert the layer widget before the coordinates widget
            if isinstance(action, QWidgetAction):
                self._layer_action = self.insertWidget(action, layer_combobox)
                self._layer_action.setVisible(False)

            action.triggered.connect(lambda _, a=action: self.logToolbarUsage(a.text()))

    @override
    def logToolbarUsage(self, action_name: str) -> None:
        logger.info(f"Plotwindow toolbar used: {action_name}")

    @override
    @Slot(bool)
    def showLayerWidget(self, show: bool) -> None:
        self._layer_action.setVisible(show)

    @override
    @Slot()
    def resetLayerWidget(
        self,
    ) -> None:
        self._layer_action.defaultWidget().setCurrentIndex(0)

    @override
    @Slot(int)
    def updateLayerWidget(self, layers: int) -> None:
        if layers != len(self._model.stringList()):
            self._model.setStringList([f"Layer {i}" for i in range(layers)])
            self.resetLayerWidget()


class PlotWidget(QWidget):
    customizationTriggered = Signal()
    layerIndexChanged = Signal(int)
    updateLayerWidget = Signal(int)
    resetLayerWidget = Signal()
    showLayerWidget = Signal(bool)

    def __init__(
        self,
        name: str,
        plotter: Union[
            "EnsemblePlot",
            "StatisticsPlot",
            "HistogramPlot",
            "GaussianKDEPlot",
            "DistributionPlot",
            "CrossEnsembleStatisticsPlot",
            "StdDevPlot",
            "ValuesOverIterationsPlot",
            "MisfitsPlot",
            "EverestGradientsPlot",
        ],
        parent: QWidget | None = None,
    ) -> None:
        QWidget.__init__(self, parent)

        self._name = name
        self._plotter = plotter
        self._figure = Figure()
        self._figure.set_layout_engine("tight")
        self._canvas = FigureCanvas(self._figure)
        self._canvas.setParent(self)
        self._canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._canvas.setFocus()

        vbox = QVBoxLayout()
        vbox.addWidget(self._canvas)
        self._toolbar = CustomNavigationToolbar(self._canvas, self)
        self._toolbar.customizationTriggered.connect(self.customizationTriggered)
        self._toolbar.layerIndexChanged.connect(self.layerIndexChanged)
        self.updateLayerWidget.connect(self._toolbar.updateLayerWidget)
        self.resetLayerWidget.connect(self._toolbar.resetLayerWidget)
        self.showLayerWidget.connect(self._toolbar.showLayerWidget)

        self._log_checkbox = QCheckBox("Log scale", self)
        self._log_checkbox.setObjectName("log_scale_checkbox")
        self._log_checkbox.setCheckable(True)
        # only for histogram plot see _sync_log_checkbox
        self._log_checkbox.setVisible(False)
        self._log_checkbox.setToolTip("Toggle data domain to log scale and back")
        self._log_checkbox.clicked.connect(self.logLogScaleButtonUsage)

        log_checkbox_row = QHBoxLayout()
        log_checkbox_row.addWidget(self._log_checkbox)
        log_checkbox_row.setContentsMargins(16, 8, 16, 8)
        log_checkbox_row.addStretch()
        vbox.addLayout(log_checkbox_row)
        vbox.addWidget(self._toolbar)
        vbox.addSpacing(8)
        self.setLayout(vbox)

        self._negative_values_in_data = False
        self._dirty = True
        self._active = False
        self.resetPlot()

    @property
    def plotUpdateRequested(self) -> pyqtBoundSignal:
        return self._log_checkbox.toggled

    def resetPlot(self) -> None:
        self._figure.clear()

    def _sync_log_checkbox(self) -> None:
        if (
            type(self._plotter).__name__
            in {
                "HistogramPlot",
                "DistributionPlot",
                "GaussianKDEPlot",
            }
            and self._negative_values_in_data is False
        ):
            self._log_checkbox.setVisible(True)
        else:
            self._log_checkbox.setVisible(False)

    @property
    def name(self) -> str:
        return self._name

    def logLogScaleButtonUsage(self) -> None:
        logger.info(f"Plotwidget utility used: 'Log scale button' in tab '{self.name}'")
        self._log_checkbox.clicked.disconnect()  # Log only once

    def updatePlot(
        self,
        plot_context: "PlotContext",
        ensemble_to_data_map: dict[EnsembleObject, pd.DataFrame],
        observations: pd.DataFrame,
        std_dev_images: dict[str, npt.NDArray[np.float32]],
        key_def: PlotApiKeyDefinition | None = None,
    ) -> None:
        self.resetPlot()
        try:
            self._sync_log_checkbox()
            plot_context.log_scale = (
                self._log_checkbox.isVisible()
                and self._log_checkbox.isChecked()
                and self._negative_values_in_data is False
            )
            self._plotter.plot(
                self._figure,
                plot_context,
                ensemble_to_data_map,
                observations,
                std_dev_images,
                key_def,
            )
            self._canvas.draw()
        except Exception as e:
            logger.exception(e)
            exc_type, _, exc_tb = sys.exc_info()
            sys.stderr.write("-" * 80 + "\n")
            traceback.print_tb(exc_tb)
            if exc_type is not None:
                sys.stderr.write(f"Exception type: {exc_type.__name__}\n")
            sys.stderr.write(f"{e}\n")
            sys.stderr.write("-" * 80 + "\n")
            sys.stderr.write(
                "An error occurred during plotting. "
                "This stack trace is helpful for diagnosing the problem."
            )
