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
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from .plot_api import EnsembleObject
from .plottery.plots.plot_tools import ConditionalAxisFormatter

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

        self._log_checkbox = QCheckBox("Log scale", self)
        self._log_checkbox.setCheckable(True)
        self._log_checkbox.setVisible(False)  # only for supported plots
        self._log_checkbox.setToolTip("Toggle data domain to log scale and back")
        self._log_checkbox.toggled.connect(self._on_log_toggle)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self._log_checkbox)
        btn_row.setContentsMargins(16, 8, 16, 8)
        btn_row.addStretch()
        vbox.addLayout(btn_row)
        vbox.addWidget(self._toolbar)
        vbox.addSpacing(8)
        self.setLayout(vbox)

        self._dirty = True
        self._active = False
        self.resetPlot()

    def resetPlot(self) -> None:
        self._figure.clear()

    def _log_axis_for_plotter(self) -> str | None:
        """Return 'x' or 'y' if this plotter supports log toggle, else None."""
        cls = type(self._plotter).__name__
        x_only = {"HistogramPlot", "GaussianKDEPlot"}
        y_only = {"DistributionPlot", "CrossEnsembleStatisticsPlot"}
        if cls in x_only:
            return "x"
        if cls in y_only:
            return "y"
        return None

    @Slot(bool)
    def _on_log_toggle(self, checked: bool) -> bool:
        """Called by toolbar when 'Log scale' is toggled. Returns True if applied."""
        axis = self._log_axis_for_plotter()
        if axis is None:
            return True
        try:
            for ax in self._figure.axes:
                if axis == "x":
                    if checked:
                        ax.set_xscale("log")
                    else:
                        ax.set_xscale("linear")
                        ax.xaxis.set_major_formatter(ConditionalAxisFormatter())

                elif checked:
                    ax.set_yscale("log")
                else:
                    ax.set_yscale("linear")
                    ax.yaxis.set_major_formatter(ConditionalAxisFormatter())
            self._canvas.draw_idle()
        except ValueError:
            self._log_checkbox.blockSignals(True)
            self._log_checkbox.setChecked(False)
            self._log_checkbox.blockSignals(False)
            return False
        return True

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
            axis = self._log_axis_for_plotter()
            self._log_checkbox.setVisible(axis is not None)
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
