import logging
import sys
import traceback
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, override

import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib import dates as mdates
from matplotlib.axes import Axes
from matplotlib.axis import Axis
from matplotlib.backend_bases import Event, MouseButton, MouseEvent, PickEvent
from matplotlib.backends.backend_qtagg import (  # type: ignore
    FigureCanvas,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.text import Text
from matplotlib.ticker import FixedFormatter, FixedLocator, FuncFormatter
from matplotlib.transforms import Bbox
from PyQt6.QtCore import QStringListModel, Qt
from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtCore import pyqtSlot as Slot
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QToolTip,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from ert.gui.plotting.plot_api import EnsembleObject, PlotApiKeyDefinition
from ert.gui.plotting.utils.plot_types import ObservationPlotLocations
from ert.gui.plotting.widgets.axis_limits_dialog import AxisLimitsDialog

if TYPE_CHECKING:
    from ert.gui.plotting.utils import PlotContext

logger = logging.getLogger(__name__)

AxisName = Literal["x", "y"]
AxisKey = tuple[int, AxisName]


@dataclass(frozen=True)
class _UserAxisLimits:
    minimum: float
    maximum: float
    is_date: bool


@dataclass(frozen=True)
class _UserCategoryRange:
    first: str
    last: str


class Plotter(Protocol):
    """Protocol for plot strategies used by PlotWidget."""

    dimensionality: int
    requires_observations: bool

    def plot(
        self,
        figure: Figure,
        plot_context: "PlotContext",
        ensemble_to_data_map: dict[EnsembleObject, pd.DataFrame],
        observation_data: pd.DataFrame,
        std_dev_images: dict[str, npt.NDArray[np.float32]],
        obs_loc: ObservationPlotLocations | None,
        key_def: PlotApiKeyDefinition | None = None,
    ) -> None: ...


class CustomNavigationToolbar(NavigationToolbar2QT):
    layer_index_changed = Signal(int)

    def __init__(
        self,
        canvas: FigureCanvas,
        parent: QWidget | None,
        *,
        coordinates: bool = True,
    ) -> None:
        super().__init__(canvas, parent, coordinates)  # type: ignore

        layer_combobox = QComboBox()
        self._model = QStringListModel()
        layer_combobox.setModel(self._model)
        layer_combobox.currentIndexChanged.connect(self.layer_index_changed)

        for action in self.actions():
            if str(action.text()).lower() in {"subplots", "customize"}:
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
    axisLabelEditRequested = Signal(str)
    titleEditRequested = Signal()
    layer_index_changed = Signal(int)
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
        self._figure.set_layout_engine("constrained")
        self._canvas = FigureCanvas(self._figure)
        self._canvas.mpl_connect("pick_event", self._on_canvas_pick)
        self._canvas.mpl_connect("motion_notify_event", self._on_canvas_motion)
        self._canvas.mpl_connect("figure_leave_event", self._on_canvas_leave)
        self._canvas.mpl_connect("button_press_event", self._on_canvas_press)
        self._canvas.setParent(self)
        self._canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._canvas.setFocus()
        self._hovered_text_artist: Text | None = None
        self._hovered_font_properties: FontProperties | None = None
        self._hovered_limits_axis: AxisKey | None = None
        self._user_axis_limits: dict[AxisKey, _UserAxisLimits | _UserCategoryRange] = {}
        self._default_axis_limits: dict[AxisKey, tuple[float, float]] = {}

        vbox = QVBoxLayout()
        vbox.addWidget(self._canvas)
        self._toolbar = CustomNavigationToolbar(self._canvas, self)
        self._toolbar.layer_index_changed.connect(self.layer_index_changed)
        self.updateLayerWidget.connect(self._toolbar.updateLayerWidget)
        self.resetLayerWidget.connect(self._toolbar.resetLayerWidget)
        self.showLayerWidget.connect(self._toolbar.showLayerWidget)

        vbox.addWidget(self._toolbar)
        vbox.addSpacing(8)
        self.setLayout(vbox)

        self._log_scale_valid_values = True
        self.resetPlot()

    def resetPlot(self) -> None:
        # Some figures contain twinaxes
        # Resetting the xscale to linear for all axes
        # to avoid log scale issues when re-plotting after a log scale plot
        for ax in self._figure.axes:
            ax.set_xscale("linear")
        self._figure.clear()

    @property
    def name(self) -> str:
        return self._name

    def update_plot(
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
            self._apply_user_axis_limits()
            self._enable_text_picking()
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

    def _enable_text_picking(self) -> None:
        self._hovered_text_artist = None
        self._hovered_font_properties = None
        QToolTip.hideText()

        for text_artist in self._editable_text_artists():
            text_artist.set_picker(True)

    def _editable_text_artists(self) -> list[Text]:
        figure_title = self._figure.get_suptitle()
        titles = [
            text_artist
            for text_artist in self._figure.texts
            if figure_title and text_artist.get_text() == figure_title
        ] or [axes.title for axes in self._figure.axes]
        return [
            text_artist
            for axes in self._figure.axes
            for text_artist in (axes.xaxis.label, axes.yaxis.label)
        ] + titles

    def _clear_hovered_text_artist(self) -> None:
        if (
            self._hovered_text_artist is not None
            and self._hovered_font_properties is not None
        ):
            self._hovered_text_artist.set_fontproperties(self._hovered_font_properties)

        self._hovered_text_artist = None
        self._hovered_font_properties = None
        self._hovered_limits_axis = None
        QToolTip.hideText()

    def _editable_text_artist_at(self, event: MouseEvent) -> Text | None:
        return next(
            (
                text_artist
                for text_artist in self._editable_text_artists()
                if text_artist.contains(event)[0]
            ),
            None,
        )

    def _on_canvas_motion(self, event: MouseEvent) -> None:
        hovered_text_artist = self._editable_text_artist_at(event)
        hovered_limits_axis = (
            self._limits_axis_at(event) if hovered_text_artist is None else None
        )
        if (
            hovered_text_artist is self._hovered_text_artist
            and hovered_limits_axis == self._hovered_limits_axis
        ):
            return

        self._clear_hovered_text_artist()
        if hovered_text_artist is None:
            if hovered_limits_axis is not None:
                self._hovered_limits_axis = hovered_limits_axis
                QToolTip.showText(
                    QCursor.pos(), "Click to set axis limits", self._canvas
                )
            self._canvas.draw_idle()
            return

        self._hovered_text_artist = hovered_text_artist
        self._hovered_font_properties = hovered_text_artist.get_fontproperties().copy()
        hovered_text_artist.set_fontweight("bold")
        QToolTip.showText(QCursor.pos(), "Click to edit", self._canvas)

        self._canvas.draw_idle()

    def _on_canvas_leave(self, _: Event) -> None:
        self._clear_hovered_text_artist()
        self._canvas.draw_idle()

    def _on_canvas_pick(self, event: PickEvent) -> None:
        for axes in self._figure.axes:
            if event.artist is axes.xaxis.label:
                self.axisLabelEditRequested.emit("x")
                return

            if event.artist is axes.yaxis.label:
                self.axisLabelEditRequested.emit("y")
                return

        if event.artist in self._editable_text_artists():
            self.titleEditRequested.emit()

    def _on_canvas_press(self, event: Event) -> None:
        if (
            not isinstance(event, MouseEvent)
            or event.button != MouseButton.LEFT
            or self._editable_text_artist_at(event) is not None
        ):
            return
        limits_axis = self._limits_axis_at(event)
        if limits_axis is not None:
            self._edit_axis_limits(*limits_axis)

    @staticmethod
    def _supports_axis_limits(axes: Axes) -> bool:
        return axes.get_visible() and axes.get_label() != "<colorbar>"

    @staticmethod
    def _axis(axes: Axes, axis_name: AxisName) -> Axis:
        return axes.xaxis if axis_name == "x" else axes.yaxis

    @staticmethod
    def _is_date_axis(axis: Axis) -> bool:
        return isinstance(axis.get_major_locator(), mdates.DateLocator)

    @staticmethod
    def _categories(axis: Axis) -> list[tuple[float, str]]:
        """(position, label) of each labelled tick on an axis whose tick labels
        are set explicitly, such as an axis of observation indices or
        ensemble names.
        """
        formatter = axis.get_major_formatter()
        if not isinstance(axis.get_major_locator(), FixedLocator) or not isinstance(
            formatter, FixedFormatter | FuncFormatter
        ):
            return []
        positions = [float(position) for position in axis.get_majorticklocs()]
        return sorted(
            (position, label)
            for position, label in zip(
                positions, formatter.format_ticks(positions), strict=True
            )
            if label
        )

    def _set_category_range(
        self,
        axes: Axes,
        axis_name: AxisName,
        categories: list[tuple[float, str]],
        first: int,
        last: int,
    ) -> None:
        positions = [position for position, _ in categories]
        half_spacing = min(np.diff(positions)) / 2 if len(positions) > 1 else 0.5
        self._set_axis_limits(
            axes,
            axis_name,
            positions[first] - half_spacing,
            positions[last] + half_spacing,
        )

    @staticmethod
    def _get_axis_limits(axes: Axes, axis_name: AxisName) -> tuple[float, float]:
        return axes.get_xlim() if axis_name == "x" else axes.get_ylim()

    def _set_axis_limits(
        self, axes: Axes, axis_name: AxisName, minimum: float, maximum: float
    ) -> None:
        limits = (
            (maximum, minimum)
            if self._axis(axes, axis_name).get_inverted()
            else (minimum, maximum)
        )
        if axis_name == "x":
            axes.set_xlim(limits)
        else:
            axes.set_ylim(limits)

    def _limits_axis_at(self, event: MouseEvent) -> AxisKey | None:
        renderer = self._canvas.get_renderer()
        axis_names: tuple[AxisName, AxisName] = ("x", "y")
        for index, axes in enumerate(self._figure.axes):
            if not self._supports_axis_limits(axes):
                continue
            for axis_name in axis_names:
                tick_label_extents = [
                    tick_label.get_window_extent(renderer)
                    for tick_label in self._axis(axes, axis_name).get_ticklabels(
                        which="both"
                    )
                    if tick_label.get_visible() and tick_label.get_text()
                ]
                if tick_label_extents and Bbox.union(tick_label_extents).contains(
                    event.x, event.y
                ):
                    return index, axis_name
        return None

    def _apply_user_axis_limits(self) -> None:
        self._default_axis_limits.clear()
        axis_names: tuple[AxisName, AxisName] = ("x", "y")
        for index, axes in enumerate(self._figure.axes):
            if not self._supports_axis_limits(axes):
                continue
            for axis_name in axis_names:
                key = (index, axis_name)
                self._default_axis_limits[key] = self._get_axis_limits(axes, axis_name)
                user_limits = self._user_axis_limits.get(key)
                axis = self._axis(axes, axis_name)
                categories = self._categories(axis)
                if isinstance(user_limits, _UserCategoryRange):
                    labels = [label for _, label in categories]
                    if user_limits.first in labels and user_limits.last in labels:
                        first = labels.index(user_limits.first)
                        last = len(labels) - 1 - labels[::-1].index(user_limits.last)
                        if first <= last:
                            self._set_category_range(
                                axes, axis_name, categories, first, last
                            )
                    continue
                if (
                    user_limits is None
                    or categories
                    or user_limits.is_date != self._is_date_axis(axis)
                    or (axis.get_scale() == "log" and user_limits.minimum <= 0)
                ):
                    continue
                self._set_axis_limits(
                    axes, axis_name, user_limits.minimum, user_limits.maximum
                )

    def _edit_axis_limits(self, axes_index: int, axis_name: AxisName) -> None:
        self._clear_hovered_text_artist()
        axes = self._figure.axes[axes_index]
        axis = self._axis(axes, axis_name)
        is_date = self._is_date_axis(axis)
        categories = self._categories(axis)
        dialog = AxisLimitsDialog(
            axis_name,
            self._get_axis_limits(axes, axis_name),
            is_date=is_date,
            is_log=axis.get_scale() == "log",
            categories=categories,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        key = (axes_index, axis_name)
        if dialog.reset_requested:
            self._user_axis_limits.pop(key, None)
            if (default_limits := self._default_axis_limits.get(key)) is not None:
                self._set_axis_limits(axes, axis_name, *sorted(default_limits))
            logger.info("Plotwindow axis limits reset")
        elif categories:
            first, last = (int(index) for index in dialog.limits())
            self._user_axis_limits[key] = _UserCategoryRange(
                categories[first][1], categories[last][1]
            )
            self._set_category_range(axes, axis_name, categories, first, last)
            logger.info("Plotwindow axis limits set")
        else:
            minimum, maximum = dialog.limits()
            self._user_axis_limits[key] = _UserAxisLimits(minimum, maximum, is_date)
            self._set_axis_limits(axes, axis_name, minimum, maximum)
            logger.info("Plotwindow axis limits set")
        self._canvas.draw_idle()
