import logging
import sys
import traceback
from typing import TYPE_CHECKING, Protocol, override

import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.backends.backend_qt5agg import (  # type: ignore
    FigureCanvas,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure
from PyQt6.QtCore import QStringListModel, Qt
from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtCore import pyqtSlot as Slot
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QComboBox,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from ert.gui.icon_utils import load_icon
from ert.gui.plotting.plot_api import EnsembleObject, PlotApiKeyDefinition
from ert.gui.plotting.utils.plot_types import ObservationPlotLocations

if TYPE_CHECKING:
    from ert.gui.plotting.utils import PlotContext

logger = logging.getLogger(__name__)


class Plotter(Protocol):
    """Protocol for plot strategies used by PlotWidget."""

    dimensionality: int
    requires_observations: bool

    def plot(
        self,
        figure: Figure,
        plot_context: "PlotContext",
        ensemble_to_data_map: dict[EnsembleObject, pd.DataFrame],
        observations: pd.DataFrame,
        std_dev_images: dict[str, npt.NDArray[np.float32]],
        obs_loc: ObservationPlotLocations | None,
        key_def: PlotApiKeyDefinition | None = None,
    ) -> None: ...


class CustomNavigationToolbar(NavigationToolbar2QT):
    customizationTriggered = Signal()
    layerIndexChanged = Signal(int)

    def __init__(
        self,
        canvas: FigureCanvas,
        parent: QWidget | None,
        *,
        coordinates: bool = True,
    ) -> None:
        super().__init__(canvas, parent, coordinates)  # type: ignore

        gear = load_icon("edit.svg")
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
        plotter: Plotter,
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

        vbox.addWidget(self._toolbar)
        vbox.addSpacing(8)
        self.setLayout(vbox)

        self._log_scale_valid_values = True
        self.resetPlot()

    def resetPlot(self) -> None:
        self._figure.clear()

    @property
    def name(self) -> str:
        return self._name

    def updatePlot(
        self,
        plot_context: "PlotContext",
        ensemble_to_data_map: dict[EnsembleObject, pd.DataFrame],
        observations: pd.DataFrame,
        std_dev_images: dict[str, npt.NDArray[np.float32]],
        obs_loc: ObservationPlotLocations | None,
        key_def: PlotApiKeyDefinition | None = None,
    ) -> None:
        self.resetPlot()
        try:
            self._plotter.plot(
                self._figure,
                plot_context,
                ensemble_to_data_map,
                observations,
                std_dev_images,
                obs_loc,
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
