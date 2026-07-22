from __future__ import annotations

from itertools import groupby
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from natsort import natsorted

from ert.gui.plotting.plot_api import EnsembleObject, PlotApiKeyDefinition
from ert.gui.plotting.shared_plots.generic_boxplot_with_scatter import (
    generate_legend_items,
    generate_plots,
)
from ert.gui.plotting.utils import ConditionalAxisFormatter, PlotTools
from ert.gui.plotting.utils.plot_context import PlotType
from ert.gui.utils import truncate_experiment_name, truncate_string

if TYPE_CHECKING:
    import numpy.typing as npt
    from matplotlib.figure import Figure

    from ert.gui.plotting.utils import PlotContext
    from ert.gui.plotting.utils.plot_types import ObservationPlotLocations

UPPER_PERCENTILE_FOR_WHISKERS = 90
LOWER_PERCENTILE_FOR_WHISKERS = 10


class CrossEnsembleStatisticsPlot:
    def __init__(self) -> None:
        self.dimensionality = 1
        self.requires_observations = False

    @staticmethod
    def plot(
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
        plot_context.plot_type = PlotType.BOX
        plot_context.deactivate_date_support()

        plot_context.y_axis = plot_context.VALUE_AXIS

        entries = natsorted(
            (
                (ensemble, color_index, ensemble_to_data_map[ensemble])
                for ensemble, color_index in zip(
                    plot_context.ensembles(),
                    plot_context.ensembles_color_indexes(),
                    strict=False,
                )
            ),
            key=lambda entry: (entry[0].experiment_name, entry[0].name),
        )
        grouped_experiments = [
            (experiment_name, list(group))
            for experiment_name, group in groupby(
                entries, key=lambda entry: entry[0].experiment_name
            )
        ]
        multiple_experiments = len(grouped_experiments) > 1
        if multiple_experiments:
            group_start = 0
            for group_index, (_, group) in enumerate(grouped_experiments):
                group_size = len(group)
                group_end = group_start + group_size

                if group_index % 2 == 1:
                    axes.axvspan(
                        group_start - 0.5,
                        group_end - 0.5,
                        color="grey",
                        alpha=0.07,
                        zorder=0,
                    )

                group_start = group_end
        box_width = 0.8
        for ensemble_index, (ensemble, color_index, data) in enumerate(entries):
            config.set_current_color(color_index)
            color = config.current_color()
            if data.empty:
                continue
            numeric_data = _assert_numeric(data)
            if numeric_data is None:
                continue
            generate_plots(
                plot_context,
                axes,
                [numeric_data.to_numpy()],
                [ensemble_index],
                box_width,
                color,
                LOWER_PERCENTILE_FOR_WHISKERS,
                UPPER_PERCENTILE_FOR_WHISKERS,
                legend_label=(
                    (
                        f"{truncate_string(ensemble.experiment_name, 20)}"
                        f" : {ensemble.name}"
                    )
                    if multiple_experiments
                    else ensemble.name
                ),
            )
        generate_legend_items(
            plot_context,
            LOWER_PERCENTILE_FOR_WHISKERS,
            UPPER_PERCENTILE_FOR_WHISKERS,
        )

        axes.set_xticks([-1, *range(len(entries)), len(entries)])

        rotation = 0
        if len(entries) > 3:
            rotation = 30

        axes.set_xticklabels(
            [""]
            + [
                (
                    f"{truncate_experiment_name(ensemble.experiment_name)}"
                    f" : {ensemble.name}"
                )
                for ensemble, _, _ in entries
            ]
            + [""],
            rotation=rotation,
        )
        axes.yaxis.set_major_formatter(ConditionalAxisFormatter())
        PlotTools.finalizePlot(
            plot_context,
            figure,
            axes,
            default_x_label="Ensemble",
            default_y_label="Value",
        )


def _assert_numeric(data: pd.DataFrame) -> pd.Series | None:
    data_series = data[0]
    if not pd.api.types.is_numeric_dtype(data_series):
        data_series = pd.to_numeric(data_series, errors="coerce")

    if not pd.api.types.is_numeric_dtype(data_series):
        data_series = None
    return data_series
