from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ert.gui.plotting.utils.plot_tools import ConditionalAxisFormatter, PlotTools

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ert.gui.plotting.utils import PlotContext


def plot_numeric_histogram(
    data: pd.Series,
    plot_context: PlotContext,
    axes: Axes,
) -> None:
    if PlotTools.array_is_empty_or_non_numeric(data):
        return

    config = plot_context.plotConfig()
    bins: str | Sequence[float]
    if plot_context.log_scale:
        log_edges = np.histogram_bin_edges(np.log10(data), bins="sqrt")
        edges = 10**log_edges
        # 10 ** log10(x) may not round-trip, which would drop the extreme values
        edges[0] = min(edges[0], data.min())
        edges[-1] = max(edges[-1], data.max())
        bins = edges.tolist()
        axes.set_xscale("log")
    else:
        bins = "sqrt"
        axes.set_xscale("linear")
        axes.xaxis.set_major_formatter(ConditionalAxisFormatter())

    axes.hist(
        data,
        bins=bins,
        alpha=0.3,
        color=config.current_color(),
    )
