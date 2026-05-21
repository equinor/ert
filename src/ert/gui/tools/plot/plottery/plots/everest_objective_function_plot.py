from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from matplotlib.ticker import MaxNLocator

from ert.gui.tools.plot.plottery.plot_context import PlotType
from ert.gui.utils import LEGEND_THRESHOLD

from .plot_tools import PlotTools

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    from matplotlib.figure import Figure

    from ert.gui.tools.plot.plot_api import EnsembleObject, PlotApiKeyDefinition
    from ert.gui.tools.plot.plottery import PlotContext


class EverestObjectiveFunctionPlot:
    """Plot of each realization's objective function value over iterations.

    Layout:
        X-axis: iteration
        Y-axis: objective value (mean or stddev)
        Glyphs: One line chart per realization, with a dot at each iteration.

    Input data: assumed to be a dictionary of DataFrames, where
    each DataFrame contains columns for 'batch_id', 'realization', and
    the objective value (e.g. 'distance').
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
        plot_context.plot_type = PlotType.LINE
        plot_context.deactivate_date_support()

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

        tooltip_data = []
        tooltip_labels = []
        color = config.next_color()
        for realization in realizations:
            data = combined[combined["realization"] == realization].sort_values(
                "batch_id"
            )
            if data.empty:
                continue

            lines = axes.plot(
                data["batch_id"],
                data[value_col],
                "-o",
                color=color,
                markersize=4,
            )
            tooltip_data.append(lines)
            tooltip_labels.append(f"Realization {int(realization)}")
            if len(realizations) <= LEGEND_THRESHOLD:
                color = config.next_color()
                config.add_legend_item(f"Realization {int(realization)}", lines[0])

        axes.xaxis.set_major_locator(MaxNLocator(integer=True))

        PlotTools.labels_on_hover(
            axes,
            plot_context,
            figure,
            data=tooltip_data,
            labels=tooltip_labels,
            disable_values=True,
            hover_color=config.next_color()[0],
        )

        PlotTools.finalizePlot(
            plot_context,
            figure,
            axes,
            default_x_label="Batch iteration",
            default_y_label="Objective value",
        )
