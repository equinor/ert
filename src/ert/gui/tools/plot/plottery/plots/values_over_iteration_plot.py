from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from .plot_tools import PlotTools

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from ert.gui.tools.plot.plot_api import EnsembleObject, PlotApiKeyDefinition
    from ert.gui.tools.plot.plottery import PlotContext


class ValuesOverIterationsPlot:
    """Plot of a value for each realization over iterations.

    Layout:
        X-axis: iteration
        Y-axis: value
        Glyphs: One line chart per realization, with a dot at each iteration.

    Input data: assumed to be a dictionary of DataFrames, where
    each DataFrame contains columns for 'batch_id', 'realization', and
    exactly one other column (named according to its source data,
    i.e., objective or control name) which is treated as the value to plot.
    """

    LEGEND_THRESHOLD = 5

    def __init__(self) -> None:
        self.dimensionality = 2
        self.requires_observations = False
        self._axes: Axes | None = None
        self.is_improvement = False
        self._legend_count = 0

    def update_legend(self, line: Line2D) -> None:
        if (
            self._axes
            and not self.is_improvement
            and self._legend_count > ValuesOverIterationsPlot.LEGEND_THRESHOLD
        ):
            self._axes.legend(handles=[line], labels=[line.get_label()])

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
        self.is_improvement = False
        config = plot_context.plotConfig()
        self._axes = figure.add_subplot(111)

        plot_context.y_axis = plot_context.VALUE_AXIS
        plot_context.x_axis = plot_context.INDEX_AXIS
        plot_context.deactivateDateSupport()

        all_dfs = [df for df in ensemble_to_data_map.values() if not df.empty]

        if not all_dfs:
            return

        combined = pd.concat(all_dfs, ignore_index=True)

        if "is_improvement" in combined.columns:
            self.is_improvement = True
            value_col = next(
                c
                for c in combined.columns
                if c not in {"batch_id", "realization", "is_improvement"}
            )
            data = combined.sort_values("batch_id")

            color = config.nextColor()
            improvement_data = data[data["is_improvement"]]

            lines = self._axes.plot(
                improvement_data["batch_id"],
                improvement_data[value_col],
                "-",
                color=color,
            )

            colors = [
                "red" if not row.is_improvement else color for _, row in data.iterrows()
            ]
            self._axes.scatter(
                data["batch_id"], data[value_col], c=colors, s=20, zorder=5
            )

            config.addLegendItem("Accepted", lines[0])
            config.addLegendItem(
                "Rejected",
                Line2D(
                    [0], [0], marker="o", color="w", markerfacecolor="red", markersize=8
                ),
            )
            self._axes.xaxis.set_major_locator(MaxNLocator(integer=True))

            PlotTools.finalizePlot(
                plot_context,
                figure,
                self._axes,
                default_x_label="Iteration",
                default_y_label="Value",
            )
            figure.tight_layout()
            return

        # Assume only one value is in the input data to make it same across
        # controls, constraints, and objectives
        value_col = next(
            c for c in combined.columns if c not in {"batch_id", "realization"}
        )

        realizations = sorted(combined["realization"].unique())
        self._legend_count = len(realizations)

        # This loop is the reason batch controls
        # plot multiple identical plots for each realization.
        for realization in realizations:
            data = combined[combined["realization"] == realization].sort_values(
                "batch_id"
            )
            if data.empty:
                continue

            color = config.nextColor()
            lines = self._axes.plot(
                data["batch_id"],
                data[value_col],
                "-o",
                label=f"Realization {int(realization)}",
                color=color,
                markersize=4,
            )
            if self._legend_count <= ValuesOverIterationsPlot.LEGEND_THRESHOLD:
                config.addLegendItem(f"Realization {int(realization)}", lines[0])

        self._axes.xaxis.set_major_locator(MaxNLocator(integer=True))
        PlotTools.finalizePlot(
            plot_context,
            figure,
            self._axes,
            default_x_label="Iteration",
            default_y_label="Value",
        )

        figure.tight_layout()
