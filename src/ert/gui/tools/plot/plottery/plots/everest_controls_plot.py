from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from matplotlib.ticker import MaxNLocator

from .plot_tools import PlotTools

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    from matplotlib.figure import Figure

    from ert.gui.tools.plot.plot_api import EnsembleObject, PlotApiKeyDefinition
    from ert.gui.tools.plot.plottery import PlotContext


class EverestControlsPlot:
    """Plot of each batch's control values over iterations.

    Layout:
        X-axis: iteration
        Y-axis: control value
        Glyphs: One line chart per control, with a dot at each iteration.

    Input data: assumed to be a dictionary of DataFrames, where
    each DataFrame contains columns for 'batch_id', 'realization', 'control_name', and
    'control_value'.
    """

    def __init__(self) -> None:
        self.dimensionality = 2
        self.requires_observations = False
        self.selected_controls: list[str] = []
        self.LEGEND_THRESHOLD = 5

    def set_selected_controls(self, controls: list[str]) -> None:
        self.selected_controls = controls

    def plot(
        self,
        figure: Figure,
        plot_context: PlotContext,
        ensemble_to_data_map: dict[EnsembleObject, pd.DataFrame],
        observation_data: pd.DataFrame,
        std_dev_images: dict[str, npt.NDArray[np.float32]],
        obs_loc: npt.NDArray[np.float32] | None,
        key_def: PlotApiKeyDefinition | None = None,
    ) -> None:
        config = plot_context.plotConfig()
        axes = figure.add_subplot(111)

        plot_context.y_axis = plot_context.VALUE_AXIS
        plot_context.x_axis = plot_context.INDEX_AXIS
        plot_context.deactivateDateSupport()

        all_dfs = [df for df in ensemble_to_data_map.values() if not df.empty]

        if not all_dfs:
            return

        combined = pd.concat(all_dfs, ignore_index=True)

        n_colors = config.getNumberOfColors()
        for i, control in (
            enumerate(self.selected_controls) if self.selected_controls else []
        ):
            data = combined[combined["control_name"] == control].sort_values("batch_id")
            if data.empty:
                continue

            color = config.nextColor()
            if i < n_colors:
                style = "-o"
            elif i < n_colors * 2:
                style = "--o"
            elif i < n_colors * 3:
                style = ":o"
            else:
                style = "-.o"
            lines = axes.plot(
                data["batch_id"],
                data["control_value"],
                style,
                color=color,
                markersize=4,
            )
            if len(self.selected_controls) <= self.LEGEND_THRESHOLD:
                config.addLegendItem(control, lines[0])
            if len(control) <= 20:
                axes.annotate(
                    control,
                    xy=(data["batch_id"].iloc[-1], data["control_value"].iloc[-1]),
                    xytext=(5, 0),
                    textcoords="offset points",
                    color=color,
                    verticalalignment="bottom",
                    horizontalalignment="left",
                )

        axes.spines["right"].set_visible(False)
        axes.spines["left"].set_visible(False)
        axes.spines["top"].set_visible(False)

        # Conversion to integer ticks for batch_id (number of batches)
        axes.xaxis.set_major_locator(MaxNLocator(integer=True))

        PlotTools.finalizePlot(
            plot_context,
            figure,
            axes,
            default_x_label="Batch Iteration",
            default_y_label="Control Value",
        )
        figure.tight_layout()
