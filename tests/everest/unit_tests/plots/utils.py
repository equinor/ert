from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent


def move_cursor(axes: Axes, x: float, y: float) -> None:
    t = axes.transData
    MouseEvent(
        "motion_notify_event",
        axes.figure.canvas,
        *t.transform((x, y)),
    )._process()  # type: ignore
