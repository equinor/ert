from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from ert.gui.plotting.plot_api import EnsembleObject, PlotApiKeyDefinition
from ert.gui.plotting.utils import ConditionalAxisFormatter, PlotTools
from ert.gui.plotting.utils.plot_context import PlotType
from ert.gui.utils import truncate_experiment_name

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

        ensemble_list = plot_context.ensembles()
        ensemble_indexes = []
        n_ens = len(ensemble_list)
        box_width = 0.8 / n_ens
        for (ensemble_index, data), color_index in zip(
            enumerate(ensemble_to_data_map.values()),
            plot_context.ensembles_color_indexes(),
            strict=False,
        ):
            config.set_current_color(color_index)
            ensemble_indexes.append(ensemble_index)
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

            config.add_legend_item(
                ensemble_list[ensemble_index].name,
                Line2D(
                    [],
                    [],
                    marker="s",
                    linestyle="None",
                    color=color,
                    label=ensemble_list[ensemble_index].name,
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

        axes.set_xticks([-1, *ensemble_indexes, len(ensemble_indexes)])

        rotation = 0
        if len(ensemble_list) > 3:
            rotation = 30

        axes.set_xticklabels(
            [""]
            + [
                (
                    f"{truncate_experiment_name(ensemble.experiment_name)}"
                    f" : {ensemble.name}"
                )
                for ensemble in ensemble_list
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
