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
from PyQt6.QtCore import QStringListModel, Qt
from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtCore import pyqtSlot as Slot
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtWidgets import QComboBox, QVBoxLayout, QWidget, QWidgetAction

from .plot_api import EnsembleObject

if TYPE_CHECKING:
    from .plottery import PlotContext
    from .plottery.plots.cesp import CrossEnsembleStatisticsPlot
    from .plottery.plots.distribution import DistributionPlot
    from .plottery.plots.ensemble import EnsemblePlot
    from .plottery.plots.gaussian_kde import GaussianKDEPlot
    from .plottery.plots.histogram import HistogramPlot
    from .plottery.plots.statistics import StatisticsPlot
    from .plottery.plots.std_dev import StdDevPlot


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

        self.addSeparator()

        self._xlog_action = QAction("X: log", self)
        self._xlog_action.setCheckable(True)
        self._xlog_action.toggled.connect(self._toggle_xscale)
        self.addAction(self._xlog_action)

        self._ylog_action = QAction("Y: log", self)
        self._ylog_action.setCheckable(True)
        self._ylog_action.toggled.connect(self._toggle_yscale)
        self.addAction(self._ylog_action)

    @Slot(bool)
    def _toggle_xscale(self, checked: bool) -> None:
        fig = self.canvas.figure
        try:
            for ax in fig.axes:
                ax.set_xscale("log" if checked else "linear")
            fig.canvas.draw_idle()
        except ValueError:
            # Non-positive data for log: revert toggle
            self._xlog_action.blockSignals(True)
            self._xlog_action.setChecked(False)
            self._xlog_action.blockSignals(False)

    @Slot(bool)
    def _toggle_yscale(self, checked: bool) -> None:
        fig = self.canvas.figure
        try:
            for ax in fig.axes:
                ax.set_yscale("log" if checked else "linear")
            fig.canvas.draw_idle()
        except ValueError:
            self._ylog_action.blockSignals(True)
            self._ylog_action.setChecked(False)
            self._ylog_action.blockSignals(False)

    def _set_checked_safely(self, action: QAction, checked: bool) -> None:
        action.blockSignals(True)
        action.setChecked(checked)
        action.blockSignals(False)

    @Slot()
    def syncScaleButtons(self) -> None:
        axes = self.canvas.figure.axes
        if not axes:
            # Default to linear when nothing is drawn
            self._set_checked_safely(self._xlog_action, False)
            self._set_checked_safely(self._ylog_action, False)
            return
        ax0 = axes[0]
        self._set_checked_safely(self._xlog_action, ax0.get_xscale() == "log")
        self._set_checked_safely(self._ylog_action, ax0.get_yscale() == "log")

    @Slot(bool)
    def showLayerWidget(self, show: bool) -> None:
        self._layer_action.setVisible(show)

    @Slot()
    def resetLayerWidget(
        self,
    ) -> None:
        self._layer_action.defaultWidget().setCurrentIndex(0)

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

        vbox.addWidget(self._toolbar)
        self.setLayout(vbox)

        self._dirty = True
        self._active = False
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
    ) -> None:
        self.resetPlot()
        try:
            self._plotter.plot(
                self._figure,
                plot_context,
                ensemble_to_data_map,
                observations,
                std_dev_images,
            )
            self._canvas.draw()
            self._toolbar.syncScaleButtons()
        except Exception as e:
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
