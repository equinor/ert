import pandas as pd
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent
from matplotlib.figure import Figure

from ert.gui.plotting.plot_api import EnsembleObject
from ert.gui.plotting.utils.plot_context import PlotContext


def move_cursor(axes: Axes, x: float, y: float) -> None:
    t = axes.transData
    MouseEvent(
        "motion_notify_event",
        axes.figure.canvas,
        *t.transform((x, y)),
    )._process()  # type: ignore


def create_everest_figure(
    plot,
    data: pd.DataFrame,
    generic_plot_context: PlotContext,
    ensemble: EnsembleObject,
) -> Figure:
    figure = Figure()
    plot.plot(
        figure,
        generic_plot_context,
        {ensemble: data},
        pd.DataFrame(),
        {},
        None,
        None,
    )
    return figure
