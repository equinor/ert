from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from matplotlib.backend_bases import Event
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Annotation
from matplotlib.typing import ColorType

T = TypeVar("T")


class ToolTipManager(ABC, Generic[T]):
    @abstractmethod
    def on_enter(self, shape: T, label: str, event: Event) -> None:
        """Method to call whenever the cursor intersects with a shape."""
        ...

    @abstractmethod
    def on_exit(self, *, redraw: bool = True) -> None:
        """Method to call whenever the cursor leaves a shape."""
        ...


class BarTooltipManager(ToolTipManager[Rectangle]):
    def __init__(self, hover_box: Annotation, figure: Figure) -> None:
        self.hover_box = hover_box
        self.figure = figure

    def on_enter(self, shape: Rectangle, label: str, event: Event) -> None:
        self.hover_box.xy = (event.xdata, event.ydata)  # type: ignore
        self.hover_box.set_text(label)
        self.hover_box.set_visible(True)
        self.figure.canvas.draw_idle()

    def on_exit(self, *, redraw: bool = True) -> None:
        self.hover_box.set_visible(False)
        self.hover_box.set_text("")
        if redraw:
            self.figure.canvas.draw_idle()


class LineTooltipManager(ToolTipManager[Line2D]):
    def __init__(
        self,
        hover_box: Annotation,
        figure: Figure,
        hover_color: str | None = None,
    ) -> None:
        self.current_line: Line2D | None = None
        self.color: ColorType | None = None
        self.alpha: float | None = None
        self.line_width: float | None = None
        self.hover_box = hover_box
        self.hover_color = hover_color
        self.figure = figure

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
        self.hover_box.set_text("")

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
