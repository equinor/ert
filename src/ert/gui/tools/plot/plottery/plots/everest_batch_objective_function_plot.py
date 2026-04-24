from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from .plot_tools import PlotTools

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    from matplotlib.figure import Figure

    from ert.gui.tools.plot.plot_api import EnsembleObject, PlotApiKeyDefinition
    from ert.gui.tools.plot.plottery import PlotContext


class EverestBatchObjectiveFunctionPlot:
    """Plot of each batch's aggregated objective function value.

    Layout:
        X-axis: iteration
        Y-axis: aggregated objective value across all realizations in the batch
        Glyphs: One line chart with a dot at each batch iteration.

    Input data: assumed to be a dictionary of DataFrames, where
    each DataFrame contains columns for 'batch_id',
    some variation of the aggregated objective value
    and 'is_improvement'.
    """

    def __init__(self) -> None:
        self.dimensionality = 2
        self.requires_observations = False

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

        value_col = next(
            c
            for c in combined.columns
            if c
            not in {
                "batch_id",
                "realization",
                "is_improvement",
                "constraint_violation_type",
            }
        )
        data = combined.sort_values("batch_id")

        color = config.nextColor()

        improvement_data = data[data["is_improvement"]]

        lines = axes.plot(
            improvement_data["batch_id"],
            improvement_data[value_col],
            "-o",
            color=color,
        )

        rejected_data = data[~data["is_improvement"]]
        if not rejected_data.empty:
            axes.scatter(
                rejected_data["batch_id"],
                rejected_data[value_col],
                c="red",
                s=20,
                zorder=5,
            )
            for _, row in rejected_data.iterrows():
                batch_id = int(row["batch_id"])
                val_type = row.get("constraint_violation_type", "N/A")
                val = row.get("constraint_violation_value", float("nan"))

                axes.annotate(
                    f"Batch {batch_id}\nViolation value: {val:.3g}\nType: {val_type}"
                    if plot_context.extended_plot_information
                    else f"Batch {batch_id}",
                    xy=(row["batch_id"], row[value_col]),
                    xytext=(5, -5),
                    textcoords="offset points",
                    color="red",
                    verticalalignment="top",
                    horizontalalignment="left",
                )
                if plot_context.extended_plot_information:
                    axes.margins(y=0.2, x=0.1)

        annotation_color = {True: color, False: "red"}
        for _, row in improvement_data.iterrows():
            improvement_value = (
                0
                if row.get("improvement_value", float("nan")) == float("-inf")
                else row.get("improvement_value", float("nan"))
            )
            batch_id = int(row["batch_id"])
            axes.annotate(
                f"Batch {batch_id}\nImprovement value: {improvement_value:.3g}"
                if plot_context.extended_plot_information
                else f"Batch {batch_id}",
                xy=(row["batch_id"], row[value_col]),
                xytext=(5, -5),
                textcoords="offset points",
                color=annotation_color[row["is_improvement"]],
                verticalalignment="top",
                horizontalalignment="left",
            )
            if plot_context.extended_plot_information:
                axes.margins(y=0.2, x=0.1)

        config.addLegendItem("Accepted", lines[0])
        config.addLegendItem(
            "Rejected",
            Line2D(
                [0], [0], marker="o", color="w", markerfacecolor="red", markersize=8
            ),
        )

        axes.margins(y=0.2, x=0.1)

        for spine in ("right", "left", "top"):
            axes.spines[spine].set_visible(False)

        axes.xaxis.set_major_locator(MaxNLocator(integer=True))

        PlotTools.finalizePlot(
            plot_context,
            figure,
            axes,
            default_x_label="Batch iteration",
            default_y_label="Aggregated objective value",
        )
        figure.tight_layout()
        return
