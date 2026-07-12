import math
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import cast

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent
from matplotlib.collections import PathCollection
from matplotlib.container import BarContainer
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Annotation
from matplotlib.transforms import Bbox
from matplotlib.typing import ColorType
from numpy.typing import ArrayLike

from ert.gui.plotting.utils.plot_context import PlotType
from ert.gui.utils import SIGNIFICANT_DIGITS

# A fallback bounding box with a width and height of 200px
# in case the hover box has not been rendered yet and returns a degenerate bbox.
FALL_BACK_BBOX = Bbox.from_bounds(0, 0, 200, 200)


class Point:
    """
    A simple class to represent a point in both coordinate and pixel space.

    Attributes:
        x_coord (float): The x coordinate in data space.
        y_coord (float): The y coordinate in data space.
        x (float): The x coordinate in pixel space.
        y (float): The y coordinate in pixel space.
    """

    def __init__(self, array: ArrayLike, axes: Axes) -> None:
        array = np.asarray(array)
        if array.shape != (2,):
            raise ValueError(f"Expected array of shape (2,), got {array.shape}")
        self.x_coord = array[0]
        self.y_coord = array[1]
        self.x = axes.transData.transform((self.x_coord, self.y_coord))[0]
        self.y = axes.transData.transform((self.x_coord, self.y_coord))[1]

    def pixels(self) -> tuple[float, float]:
        return self.x, self.y


class ValidatedMouseEvent:
    def __init__(self, event: MouseEvent, axes: Axes) -> None:
        if event.xdata is None or event.ydata is None:
            raise ValueError("MouseEvent must have valid xy coordinates")

        self._event = event
        self.inaxes = event.inaxes
        self.position = Point((event.xdata, event.ydata), axes)

    def as_mpl(self) -> MouseEvent:
        return self._event


class ToolTipManager[PlotDataType, ShapeType](ABC):
    PADDING = 10

    def __init__(
        self,
        data: PlotDataType,
        labels: list[str],
        hover_box: Annotation,
        figure: Figure,
        axes: Axes,
        hover_color: str | None = None,
    ) -> None:
        self.data = data
        self.labels = labels
        self.hover_box = hover_box
        self.figure = figure
        self.axes = axes
        self.hover_color = hover_color

    @abstractmethod
    def on_hover(self, event: ValidatedMouseEvent) -> None:
        """Method to attach to the plot's `motion_notify_event`."""
        ...

    @abstractmethod
    def on_enter(
        self, shape: ShapeType, label: str, event: ValidatedMouseEvent
    ) -> None:
        """Method to call whenever the cursor intersects with a shape."""
        ...

    @abstractmethod
    def on_exit(self, *, redraw: bool = True) -> None:
        """Method to call whenever the cursor leaves a shape."""
        ...

    def euclidean_distance(self, p: Point, q: Point) -> float:
        """Calculate the Euclidean distance between two points."""
        return math.hypot(p.x - q.x, p.y - q.y)

    def set_hover_box_position(self, event: ValidatedMouseEvent) -> None:
        """Sets position of the hover box, ensuring it stays within the axes bounds."""
        axes_bbox = self.axes.bbox
        width, height = self.hover_box.get_window_extent().size

        if width <= 1.0 or height <= 1.0:
            width, height = FALL_BACK_BBOX.size

        x, y = event.position.pixels()
        min_x = axes_bbox.x0 + self.PADDING
        max_x = axes_bbox.x1 - width - self.PADDING
        min_y = axes_bbox.y0 + self.PADDING
        max_y = axes_bbox.y1 - height - self.PADDING

        x = min(max(x, min_x), max_x)
        y = min(max(y, min_y), max_y)

        x_data, y_data = self.axes.transData.inverted().transform((x, y))
        self.hover_box.xy = (x_data, y_data)


class BarTooltipManager(ToolTipManager[list[BarContainer], Rectangle]):
    def __init__(
        self,
        data: list[BarContainer],
        labels: list[str],
        hover_box: Annotation,
        figure: Figure,
        axes: Axes,
    ) -> None:
        bars = sum(len(b.patches) for b in data)
        if bars != len(labels):
            raise ValueError(
                f"Number of labels must equal number of bars.\n"
                f"Found {len(labels)} labels and {bars} bars."
            )

        super().__init__(
            data,
            labels,
            hover_box,
            figure,
            axes,
        )

    def on_hover(self, event: ValidatedMouseEvent) -> None:
        for bar_container_idx, bars in enumerate(self.data):
            for bar_idx, bar in enumerate(bars.patches):
                contains, _ = bar.contains(event.as_mpl())
                if contains:
                    value = bar.get_height()
                    idx = bar_idx * len(self.data) + bar_container_idx
                    self.on_enter(
                        bar,
                        f"{self.labels[idx]}\nValue: {value:.{SIGNIFICANT_DIGITS}g}",
                        event,
                    )
                    return

        self.on_exit()

    def on_enter(
        self, shape: Rectangle, label: str, event: ValidatedMouseEvent
    ) -> None:
        self.hover_box.set_text(label)
        self.set_hover_box_position(event)
        self.hover_box.set_visible(True)
        self.figure.canvas.draw_idle()

    def on_exit(self, *, redraw: bool = True) -> None:
        self.hover_box.set_visible(False)
        if redraw:
            self.figure.canvas.draw_idle()


class LineTooltipManager(ToolTipManager[list[list[Line2D]], Line2D]):
    def __init__(
        self,
        data: list[list[Line2D]],
        labels: list[str],
        hover_box: Annotation,
        figure: Figure,
        axes: Axes,
        hover_color: str | None = None,
        *,
        disable_values: bool = False,
    ) -> None:
        if len(labels) != len(data):
            raise ValueError(
                f"Number of labels must equal number of line groups.\n"
                f"Found {len(labels)} labels and {len(data)} lines."
            )
        super().__init__(
            data,
            labels,
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
        event: ValidatedMouseEvent,
    ) -> None:
        for lines, label in zip(self.data, self.labels, strict=True):
            for line in lines:
                contains, idx = line.contains(event.as_mpl())
                if contains:
                    hover_text = label
                    if not self.disable_values:
                        value = np.asarray(line.get_ydata())[idx["ind"][0]]
                        hover_text += f"\nValue: {value:.{SIGNIFICANT_DIGITS}g}"
                    self.on_enter(line, hover_text, event)
                    return

        self.on_exit()

    def on_enter(self, shape: Line2D, label: str, event: ValidatedMouseEvent) -> None:
        self.hover_box.set_text(label)
        self.set_hover_box_position(event)

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
    ToolTipManager[list[PathCollection] | PathCollection, Point]
):
    def __init__(
        self,
        data: PathCollection | list[PathCollection],
        labels: list[str],
        hover_box: Annotation,
        figure: Figure,
        axes: Axes,
    ) -> None:
        points = data if isinstance(data, list) else [data]
        num_points = sum(len(np.asarray(d.get_offsets())) for d in points)
        if num_points != len(labels):
            raise ValueError(
                f"Number of labels must equal number of points.\n"
                f"Found {len(labels)} labels and {num_points} points."
            )

        super().__init__(
            data,
            labels,
            hover_box,
            figure,
            axes,
        )

    def on_hover(
        self,
        event: ValidatedMouseEvent,
    ) -> None:
        for point, label in zip(self._get_points(self.data), self.labels, strict=True):
            # This is from Matplotlib's source code for converting plot coordinates
            # to pixels and adjust according to the figure's DPI.
            threshold = self.figure.dpi / 72.0 * 5.0
            if self.euclidean_distance(point, event.position) <= threshold:
                self.on_enter(point, label, event)
                return

        self.on_exit()

    def on_enter(
        self,
        shape: Point,
        label: str,
        event: ValidatedMouseEvent,
    ) -> None:
        self.hover_box.set_text(label)
        self.set_hover_box_position(event)
        self.hover_box.set_visible(True)
        self.figure.canvas.draw_idle()

    def on_exit(self, *, redraw: bool = True) -> None:
        self.hover_box.set_visible(False)
        if redraw:
            self.figure.canvas.draw_idle()

    def _get_points(
        self, data: PathCollection | list[PathCollection]
    ) -> Iterator[Point]:
        scatters = data if isinstance(data, list) else [data]
        for scatter in scatters:
            yield from (
                Point(offset, self.axes) for offset in np.asarray(scatter.get_offsets())
            )


def create_tooltip_manager(
    data: list[BarContainer]
    | list[list[Line2D]]
    | PathCollection
    | list[PathCollection],
    labels: list[str],
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
            if isinstance(data, list) and all(
                isinstance(d, BarContainer) for d in data
            ):
                return BarTooltipManager(
                    cast(list[BarContainer], data), labels, hover_box, figure, axes
                )
        case PlotType.LINE:
            if isinstance(data, list) and all(
                isinstance(d, list) and all(isinstance(line, Line2D) for line in d)
                for d in data
            ):
                return LineTooltipManager(
                    cast(list[list[Line2D]], data),
                    labels,
                    hover_box,
                    figure,
                    axes,
                    hover_color,
                    disable_values=disable_values,
                )
        case PlotType.SCATTER:
            if isinstance(data, PathCollection) or (
                isinstance(data, list)
                and all(isinstance(d, PathCollection) for d in data)
            ):
                return ScatterTooltipManager(
                    cast(PathCollection | list[PathCollection], data),
                    labels,
                    hover_box,
                    figure,
                    axes,
                )

    raise TypeError(f"Invalid data type {type(data).__name__} for {plot_type} plot")
