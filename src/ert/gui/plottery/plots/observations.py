from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from pandas import DataFrame

    from ert.gui.plottery import PlotConfig, PlotContext


def plotObservations(
    observation_data: DataFrame, plot_context: PlotContext, axes: Axes
) -> None:
    key = plot_context.key()
    config = plot_context.plotConfig()
    ensemble_list = plot_context.ensembles()

    if (
        config.isObservationsEnabled()
        and len(ensemble_list) > 0
        and observation_data is not None
        and not observation_data.empty
    ):
        _plotObservations(axes, config, observation_data, value_column=key)


def _plotObservations(
    axes: Axes, plot_config: PlotConfig, data: DataFrame, value_column: str
) -> None:
    """
    Observations are always plotted on top. z-order set to 1000

    Since it is not possible to apply different linestyles to the errorbar, the
    line_style / fmt is used to toggle visibility of the solid errorbar, by
    using the elinewidth parameter.
    """

    style = plot_config.observationsStyle()

    # adjusting the top and bottom bar, according to the line width/thickness
    def cap_size(line_with: float) -> float:
        return 0 if line_with == 0 else math.log(line_with, 1.2) + 3

    # line style set to 'off' toggles errorbar visibility
    if not style.line_style:
        style.width = 0

    axes.errorbar(
        x=data.loc["key_index"].values,
        y=data.loc["OBS"].values,
        yerr=data.loc["STD"].values,
        fmt=style.line_style,
        ecolor=style.color,
        color=style.color,
        capsize=cap_size(style.width),
        capthick=style.width,  # same as width/thickness on error line
        alpha=style.alpha,
        linewidth=0,
        marker=style.marker,
        ms=style.size,
        elinewidth=style.width,
        zorder=1000,
    )
