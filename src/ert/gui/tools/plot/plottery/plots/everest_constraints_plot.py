from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from .plot_tools import PlotTools

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    from matplotlib.figure import Figure

    from ert.gui.tools.plot.plot_api import EnsembleObject, PlotApiKeyDefinition
    from ert.gui.tools.plot.plottery import PlotContext


class EverestConstraintsPlot:
    """Plot of each batch's constraint value over iterations.

    Layout:
        X-axis: iteration
        Y-axis: constraint value
        Glyphs: One line chart per realization, with a dot at each iteration.

    Input data: assumed to be a dictionary of DataFrames, where
    each DataFrame contains columns for 'batch_id', 'realization',
    constraint value and upper/lower bounds.
    """

    def __init__(self) -> None:
        self.dimensionality = 2
        self.requires_observations = False
        self.LEGEND_THRESHOLD = 5

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

        # Assume only one value is in the input data to make it same across
        # controls, constraints, and objectives
        value_col = next(
            c for c in combined.columns if c not in {"batch_id", "realization"}
        )

        realizations = sorted(combined["realization"].unique())

        for realization in realizations:
            data = combined[combined["realization"] == realization].sort_values(
                "batch_id"
            )
            if data.empty:
                continue

            color = config.nextColor()
            lines = axes.plot(
                data["batch_id"],
                data[value_col],
                "-o",
                color=color,
                markersize=4,
            )
            if len(realizations) <= self.LEGEND_THRESHOLD:
                config.addLegendItem(f"Realization {int(realization)}", lines[0])

        if "lower_bound" in combined.columns or "upper_bound" in combined.columns:
            ylim = axes.get_ylim()
            config.addLegendItem(
                "Outside of bounds",
                Patch(facecolor="red", alpha=0.15, edgecolor="none"),
            )
            for bound in ["lower_bound", "upper_bound"]:
                if bound in combined.columns:
                    bound_min = bound_max = combined[bound].iloc[0]

                    if bound == "lower_bound":
                        bound_min = -abs(bound_min * 1e10)
                    else:
                        bound_max = abs(bound_max * 1e10)

                    axes.axhspan(
                        bound_min,
                        bound_max,
                        alpha=0.15,
                        color="red",
                        zorder=0,
                    )
                    axes.axhline(
                        bound_max if bound == "lower_bound" else bound_min,
                        color="grey",
                        linestyle="--",
                        alpha=0.7,
                    )
            axes.set_ylim(ylim)

        axes.xaxis.set_major_locator(MaxNLocator(integer=True))
        axes.spines["right"].set_visible(False)
        axes.spines["top"].set_visible(False)
        PlotTools.finalizePlot(
            plot_context,
            figure,
            axes,
            default_x_label="Batch iteration",
            default_y_label="Constraint value",
        )
        figure.tight_layout()
