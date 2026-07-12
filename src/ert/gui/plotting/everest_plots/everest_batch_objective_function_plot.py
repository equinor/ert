from __future__ import annotations

from textwrap import dedent
from typing import TYPE_CHECKING

import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from ert.gui.plotting.utils.plot_context import PlotType
from ert.gui.plotting.utils.plot_tools import PlotTools
from ert.gui.utils import SIGNIFICANT_DIGITS

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    from matplotlib.figure import Figure

    from ert.gui.plotting.plot_api import EnsembleObject, PlotApiKeyDefinition
    from ert.gui.plotting.utils import PlotContext
    from ert.gui.plotting.utils.plot_types import ObservationPlotLocations


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
        obs_loc: ObservationPlotLocations | None,
        key_def: PlotApiKeyDefinition | None = None,
    ) -> None:
        config = plot_context.plotConfig()
        axes = figure.add_subplot(111)

        plot_context.y_axis = plot_context.VALUE_AXIS
        plot_context.x_axis = plot_context.INDEX_AXIS
        plot_context.deactivate_date_support()
        all_dfs = [df for df in ensemble_to_data_map.values() if not df.empty]

        if not all_dfs:
            axes.text(0.5, 0.5, "No data", ha="center", va="center")
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

        color = config.next_color()

        improvement_data = data[data["is_improvement"]]

        lines = axes.plot(
            improvement_data["batch_id"],
            improvement_data[value_col],
            "-o",
            color=color,
        )

        scatter_labels = []
        scatter_data = []
        if not improvement_data.empty:
            scatter_data.append(
                axes.scatter(
                    improvement_data["batch_id"],
                    improvement_data[value_col],
                    c=color,
                    s=20,
                    zorder=5,
                )
            )
            for _, row in improvement_data.iterrows():
                batch_id = int(row["batch_id"])
                scatter_labels.append(f"Batch {batch_id}")

        rejected_data = data[~data["is_improvement"]]
        if not rejected_data.empty:
            scatter_data.append(
                axes.scatter(
                    rejected_data["batch_id"],
                    rejected_data[value_col],
                    c="red",
                    s=20,
                    zorder=5,
                )
            )
            for _, row in rejected_data.iterrows():
                batch_id = int(row["batch_id"])
                val_type = row.get("constraint_violation_type", "N/A")
                val = row.get("constraint_violation_value", float("nan"))

                scatter_labels.append(
                    dedent(
                        f"""
                        Batch {batch_id}
                        Objective value: {row[value_col]:.{SIGNIFICANT_DIGITS}g}
                        Violation value: {val:.{SIGNIFICANT_DIGITS}g}
                        Type: {val_type}
                        """
                    ).strip("\n")
                )

        config.add_legend_item("Accepted", lines[0])
        config.add_legend_item(
            "Rejected",
            Line2D(
                [0], [0], marker="o", color="w", markerfacecolor="red", markersize=8
            ),
        )

        axes.xaxis.set_major_locator(MaxNLocator(integer=True))

        if scatter_labels and scatter_data:
            PlotTools.labels_on_hover(
                PlotType.SCATTER,
                axes,
                figure,
                data=scatter_data,
                labels=scatter_labels,
            )

        PlotTools.finalizePlot(
            plot_context,
            figure,
            axes,
            default_x_label="Batch iteration",
            default_y_label="Aggregated objective value",
        )
