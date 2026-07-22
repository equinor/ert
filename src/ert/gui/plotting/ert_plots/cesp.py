from __future__ import annotations

from itertools import groupby
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from natsort import natsorted

from ert.gui.plotting.plot_api import EnsembleObject, PlotApiKeyDefinition
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
        outlier = plot_context.outliers
        scatter = plot_context.scatter_plot
        box = plot_context.box_plot
        mean = plot_context.mean
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
            if box:
                axes.boxplot(
                    numeric_data,
                    positions=[ensemble_index],
                    widths=box_width,
                    whis=(
                        LOWER_PERCENTILE_FOR_WHISKERS,
                        UPPER_PERCENTILE_FOR_WHISKERS,
                    ),
                    patch_artist=True,
                    showfliers=outlier,
                    boxprops={
                        "facecolor": color,
                        "alpha": 0.8,
                        "edgecolor": color,
                        "linewidth": 0.7,
                    },
                    whiskerprops={
                        "color": color,
                        "alpha": 1,
                        "linewidth": 1,
                        "linestyle": "--",
                    },
                    capprops={
                        "color": color,
                        "alpha": 1,
                        "linewidth": 2,
                        "linestyle": "--",
                    },
                    medianprops={"color": "black", "linewidth": 1, "alpha": 1},
                    flierprops={
                        "marker": "o",
                        "alpha": 1,
                        "markeredgewidth": 0.3 + (0.4 * (1 - box_width)),
                        "markeredgecolor": color,
                        "markerfacecolor": "none",
                    },
                )
            if scatter:
                rng = np.random.default_rng(42)
                jitter = box_width * 0.5

                x_points: list[np.ndarray] = []
                x_points.append(
                    ensemble_index
                    + rng.uniform(-jitter / 2, jitter / 2, size=len(numeric_data))
                )

                x_all = np.concatenate(x_points)

                axes.scatter(
                    x_all,
                    numeric_data,
                    color=color,
                    alpha=0.35,
                    linewidths=0,
                    zorder=2,  # above bands/boxes
                )
            if mean:
                axes.plot(
                    ensemble_index,
                    np.nanmean(numeric_data),
                    "D",
                    markersize=4,
                    color="black",
                    zorder=3,  # Above boxes and scatter
                )

            legend_label = (
                (f"{truncate_string(ensemble.experiment_name, 20)} : {ensemble.name}")
                if multiple_experiments
                else ensemble.name
            )
            config.add_legend_item(
                legend_label,
                Line2D(
                    [],
                    [],
                    marker="s",
                    linestyle="None",
                    color=color,
                    label=legend_label,
                ),
            )
        if box:
            config.add_legend_item(
                "Median", Line2D([0], [0], color="black", linewidth=0.9, alpha=1)
            )
            config.add_legend_item(
                (
                    f"Whiskers ({LOWER_PERCENTILE_FOR_WHISKERS}-"
                    f"{UPPER_PERCENTILE_FOR_WHISKERS} %)"
                ),
                Line2D([0], [0], color="black", linewidth=2, linestyle="--", alpha=1),
            )
            if outlier:
                config.add_legend_item(
                    "Outliers",
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="none",
                        markeredgecolor="black",
                        markerfacecolor="none",
                        markersize=6,
                        alpha=1,
                    ),
                )
        if scatter:
            config.add_legend_item(
                "Scatter points",
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="black",
                    markeredgecolor="None",
                    linestyle="None",
                    alpha=0.35,
                ),
            )
        if mean:
            config.add_legend_item(
                "Mean",
                Line2D(
                    [0],
                    [0],
                    marker="D",
                    color="black",
                    markersize=4,
                    linestyle="None",
                    alpha=1,
                ),
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
