from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from typing_extensions import TypedDict

from ert.gui.plotting.plot_api import EnsembleObject, PlotApiKeyDefinition
from ert.gui.plotting.utils import ConditionalAxisFormatter, PlotTools
from ert.gui.utils import truncate_experiment_name

if TYPE_CHECKING:
    import numpy.typing as npt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from pandas import DataFrame

    from ert.gui.plotting.utils import PlotConfig, PlotContext
    from ert.gui.plotting.utils.plot_types import ObservationPlotLocations


class CrossEnsembleStatisticsData(TypedDict):
    index: list[int]
    mean: dict[int, float]
    min: dict[int, float]
    max: dict[int, float]
    p10: dict[int, float]
    p33: dict[int, float]
    p50: dict[int, float]
    p67: dict[int, float]
    p90: dict[int, float]
    std: dict[int, float]


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
        plot_cross_ensemble_statistics(
            figure, plot_context, ensemble_to_data_map, observation_data
        )


def plot_cross_ensemble_statistics(
    figure: Figure,
    plot_context: PlotContext,
    ensemble_to_data_map: dict[EnsembleObject, pd.DataFrame],
    _observation_data: DataFrame,
) -> None:
    config = plot_context.plotConfig()
    axes = figure.add_subplot(111)

    plot_context.deactivate_date_support()

    plot_context.y_axis = plot_context.VALUE_AXIS

    ensemble_list = plot_context.ensembles()
    ensemble_indexes = []
    ces: CrossEnsembleStatisticsData = {
        "index": [],
        "mean": {},
        "min": {},
        "max": {},
        "std": {},
        "p10": {},
        "p33": {},
        "p50": {},
        "p67": {},
        "p90": {},
    }
    for (ensemble_index, data), color_index in zip(
        enumerate(ensemble_to_data_map.values()),
        plot_context.ensembles_color_indexes(),
        strict=False,
    ):
        config.set_current_color(color_index)
        ensemble_indexes.append(ensemble_index)
        std_dev_factor = config.get_standard_deviation_factor()

        if not data.empty:
            numeric_data = _assert_numeric(data)
            if numeric_data is not None:
                ces["index"].append(ensemble_index)
                ces["mean"][ensemble_index] = numeric_data.mean()
                ces["min"][ensemble_index] = numeric_data.min()
                ces["max"][ensemble_index] = numeric_data.max()
                ces["std"][ensemble_index] = numeric_data.std() * std_dev_factor
                ces["p10"][ensemble_index] = numeric_data.quantile(0.1)
                ces["p33"][ensemble_index] = numeric_data.quantile(0.33)
                ces["p50"][ensemble_index] = numeric_data.quantile(0.5)
                ces["p67"][ensemble_index] = numeric_data.quantile(0.67)
                ces["p90"][ensemble_index] = numeric_data.quantile(0.9)

                _plot_cross_ensemble_statistics(axes, config, ces, ensemble_index)

    if config.is_distribution_line_enabled() and len(ces["index"]) > 1:
        _plot_connection_lines(axes, config, ces)

    _add_statistics_legends(config)

    axes.set_xticks([-1, *ensemble_indexes, len(ensemble_indexes)])

    rotation = 0
    if len(ensemble_list) > 3:
        rotation = 30

    axes.set_xticklabels(
        [""]
        + [
            f"{truncate_experiment_name(ensemble.experiment_name)} : {ensemble.name}"
            for ensemble in ensemble_list
        ]
        + [""],
        rotation=rotation,
    )
    axes.yaxis.set_major_formatter(ConditionalAxisFormatter())
    PlotTools.finalizePlot(
        plot_context, figure, axes, default_x_label="Ensemble", default_y_label="Value"
    )


def _add_statistics_legends(plot_config: PlotConfig) -> None:
    _add_statistics_legend(plot_config, "mean")
    _add_statistics_legend(plot_config, "p50")
    _add_statistics_legend(plot_config, "min-max", 0.2)
    _add_statistics_legend(plot_config, "p10-p90", 0.4)
    _add_statistics_legend(plot_config, "std", 0.4)
    _add_statistics_legend(plot_config, "p33-p67", 0.6)


def _add_statistics_legend(
    plot_config: PlotConfig, style_name: str, alpha_multiplier: float = 1.0
) -> None:
    style = plot_config.get_statistics_style(style_name)
    if style.is_visible():
        if style.line_style == "#":
            rectangle = Rectangle(
                (0, 0), 1, 1, color="black", alpha=style.alpha * alpha_multiplier
            )  # creates rectangle patch for legend use.
            plot_config.add_legend_item(style.name, rectangle)
        else:
            line = Line2D(
                [],
                [],
                color="black",
                marker=style.marker,
                linestyle=style.line_style,
                linewidth=style.width,
                alpha=style.alpha,
            )
            plot_config.add_legend_item(style.name, line)


def _assert_numeric(data: pd.DataFrame) -> pd.Series | None:
    data_series = data[0]
    if not pd.api.types.is_numeric_dtype(data_series):
        data_series = pd.to_numeric(data_series, errors="coerce")

    if not pd.api.types.is_numeric_dtype(data_series):
        data_series = None
    return data_series


def _plot_cross_ensemble_statistics(
    axes: Axes, plot_config: PlotConfig, data: CrossEnsembleStatisticsData, index: int
) -> None:
    axes.set_xlabel(plot_config.x_label())  # type: ignore
    axes.set_ylabel(plot_config.y_label())  # type: ignore

    style = plot_config.get_statistics_style("mean")
    if style.is_visible():
        axes.plot(
            [index],
            data["mean"][index],
            alpha=style.alpha,
            linestyle="",
            color=style.color,
            marker=style.marker,
            markersize=style.size,
        )

    style = plot_config.get_statistics_style("p50")
    if style.is_visible():
        axes.plot(
            [index],
            data["p50"][index],
            alpha=style.alpha,
            linestyle="",
            color=style.color,
            marker=style.marker,
            markersize=style.size,
        )

    style = plot_config.get_statistics_style("std")
    if style.is_visible():
        axes.plot(
            [index],
            data["mean"][index] + data["std"][index],
            alpha=style.alpha,
            linestyle="",
            color=style.color,
            marker=style.marker,
            markersize=style.size,
        )
        axes.plot(
            [index],
            data["mean"][index] - data["std"][index],
            alpha=style.alpha,
            linestyle="",
            color=style.color,
            marker=style.marker,
            markersize=style.size,
        )

    style = plot_config.get_statistics_style("min-max")
    if style.is_visible():
        axes.plot(
            [index],
            data["min"][index],
            alpha=style.alpha,
            linestyle="",
            color=style.color,
            marker=style.marker,
            markersize=style.size,
        )
        axes.plot(
            [index],
            data["max"][index],
            alpha=style.alpha,
            linestyle="",
            color=style.color,
            marker=style.marker,
            markersize=style.size,
        )

    style = plot_config.get_statistics_style("p10-p90")
    if style.is_visible():
        axes.plot(
            [index],
            data["p10"][index],
            alpha=style.alpha,
            linestyle="",
            color=style.color,
            marker=style.marker,
            markersize=style.size,
        )
        axes.plot(
            [index],
            data["p90"][index],
            alpha=style.alpha,
            linestyle="",
            color=style.color,
            marker=style.marker,
            markersize=style.size,
        )

    style = plot_config.get_statistics_style("p33-p67")
    if style.is_visible():
        axes.plot(
            [index],
            data["p33"][index],
            alpha=style.alpha,
            linestyle="",
            color=style.color,
            marker=style.marker,
            markersize=style.size,
        )
        axes.plot(
            [index],
            data["p67"][index],
            alpha=style.alpha,
            linestyle="",
            color=style.color,
            marker=style.marker,
            markersize=style.size,
        )


def _plot_connection_lines(
    axes: Axes,
    plot_config: PlotConfig,
    ces: CrossEnsembleStatisticsData,
) -> None:
    line_style = plot_config.distributionLineStyle()
    index_list = ces["index"]
    for index in range(len(index_list) - 1):
        from_index = index_list[index]
        to_index = index_list[index + 1]

        x = [from_index, to_index]

        style = plot_config.get_statistics_style("mean")
        if style.is_visible():
            y = [ces["mean"][from_index], ces["mean"][to_index]]
            axes.plot(
                x,
                y,
                alpha=line_style.alpha,
                linestyle=style.line_style,
                color=line_style.color,
                linewidth=style.width,
            )

        style = plot_config.get_statistics_style("p50")
        if style.is_visible():
            y = [ces["p50"][from_index], ces["p50"][to_index]]
            axes.plot(
                x,
                y,
                alpha=line_style.alpha,
                linestyle=style.line_style,
                color=line_style.color,
                linewidth=style.width,
            )

        style = plot_config.get_statistics_style("std")
        if style.is_visible():
            mean = [ces["mean"][from_index], ces["mean"][to_index]]
            std = [ces["std"][from_index], ces["std"][to_index]]

            y_1 = [mean[0] + std[0], mean[1] + std[1]]
            y_2 = [mean[0] - std[0], mean[1] - std[1]]

            linestyle = style.line_style
            if linestyle == "#":
                linestyle = ""

            axes.plot(
                x,
                y_1,
                alpha=style.alpha,
                linestyle=linestyle,
                color=line_style.color,
                linewidth=style.width,
            )
            axes.plot(
                x,
                y_2,
                alpha=style.alpha,
                linestyle=linestyle,
                color=line_style.color,
                linewidth=style.width,
            )

        style = plot_config.get_statistics_style("min-max")
        if style.is_visible():
            y_1 = [ces["min"][from_index], ces["min"][to_index]]
            y_2 = [ces["max"][from_index], ces["max"][to_index]]

            linestyle = style.line_style
            if linestyle == "#":
                linestyle = ""

            axes.plot(
                x,
                y_1,
                alpha=style.alpha,
                linestyle=linestyle,
                color=line_style.color,
                linewidth=style.width,
            )
            axes.plot(
                x,
                y_2,
                alpha=style.alpha,
                linestyle=linestyle,
                color=line_style.color,
                linewidth=style.width,
            )

        style = plot_config.get_statistics_style("p10-p90")
        if style.is_visible():
            y_1 = [ces["p10"][from_index], ces["p10"][to_index]]
            y_2 = [ces["p90"][from_index], ces["p90"][to_index]]

            linestyle = style.line_style
            if linestyle == "#":
                linestyle = ""

            axes.plot(
                x,
                y_1,
                alpha=style.alpha,
                linestyle=linestyle,
                color=line_style.color,
                linewidth=style.width,
            )
            axes.plot(
                x,
                y_2,
                alpha=style.alpha,
                linestyle=linestyle,
                color=line_style.color,
                linewidth=style.width,
            )

        style = plot_config.get_statistics_style("p33-p67")
        if style.is_visible():
            y_1 = [ces["p33"][from_index], ces["p33"][to_index]]
            y_2 = [ces["p67"][from_index], ces["p67"][to_index]]

            linestyle = style.line_style
            if linestyle == "#":
                linestyle = ""

            axes.plot(
                x,
                y_1,
                alpha=style.alpha,
                linestyle=linestyle,
                color=line_style.color,
                linewidth=style.width,
            )
            axes.plot(
                x,
                y_2,
                alpha=style.alpha,
                linestyle=linestyle,
                color=line_style.color,
                linewidth=style.width,
            )
