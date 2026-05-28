import math
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backend_bases import Event
from matplotlib.collections import PathCollection
from matplotlib.container import BarContainer
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Annotation
from matplotlib.typing import ColorType

from ert.gui.tools.plot.plottery.plot_context import PlotType
from ert.gui.utils import SIGNIFICANT_DIGITS

PlotDataType = TypeVar("PlotDataType")
ShapeType = TypeVar("ShapeType")


class ToolTipManager(ABC, Generic[PlotDataType, ShapeType]):
    def __init__(
        self,
        hover_box: Annotation,
        figure: Figure,
        axes: Axes,
        hover_color: str | None = None,
    ) -> None:
        self.hover_box = hover_box
        self.figure = figure
        self.axes = axes
        self.hover_color = hover_color

    @abstractmethod
    def on_hover(self, event: Event, data: PlotDataType, labels: list[str]) -> None:
        """Method to attach to the plot's `motion_notify_event`."""
        ...

    @abstractmethod
    def on_enter(self, shape: ShapeType, label: str, event: Event) -> None:
        """Method to call whenever the cursor intersects with a shape."""
        ...

    @abstractmethod
    def on_exit(self, *, redraw: bool = True) -> None:
        """Method to call whenever the cursor leaves a shape."""
        ...

    def euclidean_distance(
        self, p: tuple[float, float], q: tuple[float, float]
    ) -> float:
        """Calculate the Euclidean distance between two points."""
        return math.hypot(p[0] - q[0], p[1] - q[1])


class BarTooltipManager(ToolTipManager[list[BarContainer], Rectangle]):
    def on_hover(
        self, event: Event, data: list[BarContainer], labels: list[str]
    ) -> None:
        if event.inaxes != self.axes:  # type: ignore
            self.on_exit()
            return
        for bar_container_idx, bars in enumerate(data):
            for bar_idx, bar in enumerate(bars.patches):
                if bar.contains(event)[0]:  # type: ignore
                    value = bar.get_height()
                    idx = bar_idx * len(data) + bar_container_idx
                    self.on_enter(
                        bar,
                        f"{labels[idx]}\nValue: {value:.{SIGNIFICANT_DIGITS}g}",
                        event,
                    )
                    return

        self.on_exit()

    def on_enter(self, shape: Rectangle, label: str, event: Event) -> None:
        self.hover_box.xy = (event.xdata, event.ydata)  # type: ignore
        self.hover_box.set_text(label)
        self.hover_box.set_visible(True)
        self.figure.canvas.draw_idle()

    def on_exit(self, *, redraw: bool = True) -> None:
        self.hover_box.set_visible(False)
        if redraw:
            self.figure.canvas.draw_idle()


class LineTooltipManager(ToolTipManager[list[list[Line2D]], Line2D]):
    def __init__(
        self,
        hover_box: Annotation,
        figure: Figure,
        axes: Axes,
        hover_color: str | None = None,
        *,
        disable_values: bool = False,
    ) -> None:
        super().__init__(
            hover_box,
            figure,
            axes,
            hover_color,
        )
        self.current_line: Line2D | None = None
        self.color: ColorType | None = None
        self.alpha: float | None = None
        self.line_width: float | None = None
        self.disable_values = disable_values

    def on_hover(
        self,
        event: Event,
        data: list[list[Line2D]],
        labels: list[str],
    ) -> None:
        if event.inaxes != self.axes:  # type: ignore
            self.on_exit()
            return

        for lines, label in zip(data, labels, strict=True):
            for line in lines:
                contains, idx = line.contains(event)  # type: ignore
                if contains:
                    hover_text = label
                    if not self.disable_values:
                        value = line.get_ydata()[idx["ind"][0]]  # type: ignore
                        hover_text += f"\nValue: {value:.{SIGNIFICANT_DIGITS}g}"  # type: ignore
                    self.on_enter(line, hover_text, event)
                    return

        self.on_exit()

    def on_enter(self, shape: Line2D, label: str, event: Event) -> None:
        self.hover_box.xy = (event.xdata, event.ydata)  # type: ignore
        self.hover_box.set_text(label)

        # Only need to update tooltip text and position if its the same line.
        if self.current_line == shape:
            self.figure.canvas.draw_idle()
            return

        self.on_exit(redraw=False)

        self.current_line = shape

        if self.hover_color is not None:
            self.color = shape.get_color()
            self.current_line.set_color(self.hover_color)

        self.alpha = shape.get_alpha()
        self.current_line.set_alpha(1.0)

        self.line_width = shape.get_linewidth()
        self.current_line.set_linewidth(self.line_width * 1.5)

        self.hover_box.set_visible(True)

        self.figure.canvas.draw_idle()

    def on_exit(self, *, redraw: bool = True) -> None:
        if self.current_line is None:
            return

        self.hover_box.set_visible(False)

        if self.hover_color is not None and self.color is not None:
            self.current_line.set_color(self.color)

        if self.alpha is not None:
            self.current_line.set_alpha(self.alpha)

        if self.line_width is not None:
            self.current_line.set_linewidth(self.line_width)

        (
            self.current_line,
            self.color,
            self.alpha,
            self.line_width,
        ) = None, None, None, None

        if redraw:
            self.figure.canvas.draw_idle()


class ScatterTooltipManager(
    ToolTipManager[PathCollection, np.ndarray[tuple[int], np.dtype[np.float64]]]
):
    def on_hover(
        self,
        event: Event,
        data: PathCollection,
        labels: list[str],
    ) -> None:
        if event.inaxes != self.axes:  # type: ignore
            self.on_exit()
            return

        scatter_points: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.asarray(
            data.get_offsets()
        )
        for scatter, label in zip(scatter_points, labels, strict=True):
            pp = self.axes.transData.transform(scatter)
            ep = self.axes.transData.transform((event.xdata, event.ydata))  # type: ignore

            # This is from Matplotlib's source code for converting plot coordinates
            # to pixels and adjust according to the figure's DPI.
            threshold = self.figure.dpi / 72.0 * 5.0
            if self.euclidean_distance((pp[0], pp[1]), (ep[0], ep[1])) <= threshold:
                self.on_enter(scatter, label, event)
                return

        self.on_exit()

    def on_enter(
        self,
        shape: np.ndarray[tuple[int], np.dtype[np.float64]],
        label: str,
        event: Event,
    ) -> None:
        self.hover_box.xy = (event.xdata, event.ydata)  # type: ignore
        self.hover_box.set_text(label)
        self.hover_box.set_visible(True)
        self.figure.canvas.draw_idle()

    def on_exit(self, *, redraw: bool = True) -> None:
        self.hover_box.set_visible(False)
        if redraw:
            self.figure.canvas.draw_idle()


def create_tooltip_manager(
    plot_type: PlotType,
    hover_box: Annotation,
    figure: Figure,
    axes: Axes,
    *,
    hover_color: str | None = None,
    disable_values: bool = False,
) -> BarTooltipManager | LineTooltipManager | ScatterTooltipManager:
    match plot_type:
        case PlotType.BAR:
            return BarTooltipManager(hover_box, figure, axes)
        case PlotType.LINE:
            return LineTooltipManager(
                hover_box, figure, axes, hover_color, disable_values=disable_values
            )
        case PlotType.SCATTER:
            return ScatterTooltipManager(hover_box, figure, axes)
