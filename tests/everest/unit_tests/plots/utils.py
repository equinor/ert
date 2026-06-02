from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent


def move_cursor(ax: Axes, x: float, y: float) -> None:
    t = ax.transData
    MouseEvent(
        "motion_notify_event",
        ax.figure.canvas,
        *t.transform((x, y)),
    )._process()  # type: ignore
