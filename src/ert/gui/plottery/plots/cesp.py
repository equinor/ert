from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, TypedDict

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from ert.gui.tools.plot.plot_api import EnsembleObject

from .plot_tools import PlotTools

if TYPE_CHECKING:
    import numpy.typing as npt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from pandas import DataFrame

    from ert.gui.plottery import PlotConfig, PlotContext


CcsData = TypedDict(
    "CcsData",
    {
        "index": List[int],
        "mean": Dict[int, float],
        "min": Dict[int, float],
        "max": Dict[int, float],
        "p10": Dict[int, float],
        "p33": Dict[int, float],
        "p50": Dict[int, float],
        "p67": Dict[int, float],
        "p90": Dict[int, float],
        "std": Dict[int, float],
    },
)


class CrossEnsembleStatisticsPlot:
    def __init__(self) -> None:
        self.dimensionality = 1

    @staticmethod
    def plot(
        figure: Figure,
        plot_context: PlotContext,
        ensemble_to_data_map: Dict[EnsembleObject, pd.DataFrame],
        observation_data: pd.DataFrame,
        std_dev_images: Dict[str, npt.NDArray[np.float32]],
    ) -> None:
        plotCrossEnsembleStatistics(
            figure, plot_context, ensemble_to_data_map, observation_data
        )


def plotCrossEnsembleStatistics(
    figure: Figure,
    plot_context: "PlotContext",
    ensemble_to_data_map: Dict[EnsembleObject, pd.DataFrame],
    _observation_data: DataFrame,
) -> None:
    config = plot_context.plotConfig()
    axes = figure.add_subplot(111)

    plot_context.deactivateDateSupport()

    plot_context.y_axis = plot_context.VALUE_AXIS

    if plot_context.log_scale:
        axes.set_yscale("log")

    ensemble_list = plot_context.ensembles()
    ensemble_indexes = []
    ccs: CcsData = {
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
    for ensemble_index, data in enumerate(ensemble_to_data_map.values()):
        ensemble_indexes.append(ensemble_index)
        std_dev_factor = config.getStandardDeviationFactor()

        if not data.empty:
            data = _assertNumeric(data)
            if data is not None:
                ccs["index"].append(ensemble_index)
                ccs["mean"][ensemble_index] = data.mean()
                ccs["min"][ensemble_index] = data.min()
                ccs["max"][ensemble_index] = data.max()
                ccs["std"][ensemble_index] = data.std() * std_dev_factor
                ccs["p10"][ensemble_index] = data.quantile(0.1)
                ccs["p33"][ensemble_index] = data.quantile(0.33)
                ccs["p50"][ensemble_index] = data.quantile(0.5)
                ccs["p67"][ensemble_index] = data.quantile(0.67)
                ccs["p90"][ensemble_index] = data.quantile(0.9)

                _plotCrossEnsembleStatistics(axes, config, ccs, ensemble_index)
                config.nextColor()

    if config.isDistributionLineEnabled() and len(ccs["index"]) > 1:
        _plotConnectionLines(axes, config, ccs)

    _addStatisticsLegends(config)

    axes.set_xticks([-1, *ensemble_indexes, len(ensemble_indexes)])

    rotation = 0
    if len(ensemble_list) > 3:
        rotation = 30

    axes.set_xticklabels(
        [""]
        + [
            f"{ensemble.experiment_name} : {ensemble.name}"
            for ensemble in ensemble_list
        ]
        + [""],
        rotation=rotation,
    )

    PlotTools.finalizePlot(
        plot_context, figure, axes, default_x_label="Ensemble", default_y_label="Value"
    )


def _addStatisticsLegends(plot_config: "PlotConfig") -> None:
    _addStatisticsLegend(plot_config, "mean")
    _addStatisticsLegend(plot_config, "p50")
    _addStatisticsLegend(plot_config, "min-max", 0.2)
    _addStatisticsLegend(plot_config, "p10-p90", 0.4)
    _addStatisticsLegend(plot_config, "std", 0.4)
    _addStatisticsLegend(plot_config, "p33-p67", 0.6)


def _addStatisticsLegend(
    plot_config: "PlotConfig", style_name: str, alpha_multiplier: float = 1.0
) -> None:
    style = plot_config.getStatisticsStyle(style_name)
    if style.isVisible():
        if style.line_style == "#":
            rectangle = Rectangle(
                (0, 0), 1, 1, color="black", alpha=style.alpha * alpha_multiplier
            )  # creates rectangle patch for legend use.
            plot_config.addLegendItem(style.name, rectangle)
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
            plot_config.addLegendItem(style.name, line)


def _assertNumeric(data: pd.DataFrame) -> pd.Series:
    data_series = data[0]
    if data_series.dtype == "object":
        try:
            data_series = pd.to_numeric(data_series, errors="coerce")
        except AttributeError:
            data_series = data_series.convert_objects(convert_numeric=True)

    if data_series.dtype == "object":
        data_series = None
    return data_series


def _plotCrossEnsembleStatistics(
    axes: "Axes", plot_config: "PlotConfig", data: CcsData, index: int
) -> None:
    axes.set_xlabel(plot_config.xLabel())  # type: ignore
    axes.set_ylabel(plot_config.yLabel())  # type: ignore

    style = plot_config.getStatisticsStyle("mean")
    if style.isVisible():
        axes.plot(
            [index],
            data["mean"][index],
            alpha=style.alpha,
            linestyle="",
            color=style.color,
            marker=style.marker,
            markersize=style.size,
        )

    style = plot_config.getStatisticsStyle("p50")
    if style.isVisible():
        axes.plot(
            [index],
            data["p50"][index],
            alpha=style.alpha,
            linestyle="",
            color=style.color,
            marker=style.marker,
            markersize=style.size,
        )

    style = plot_config.getStatisticsStyle("std")
    if style.isVisible():
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

    style = plot_config.getStatisticsStyle("min-max")
    if style.isVisible():
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

    style = plot_config.getStatisticsStyle("p10-p90")
    if style.isVisible():
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

    style = plot_config.getStatisticsStyle("p33-p67")
    if style.isVisible():
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


def _plotConnectionLines(
    axes: "Axes",
    plot_config: "PlotConfig",
    ccs: CcsData,
) -> None:
    line_style = plot_config.distributionLineStyle()
    index_list = ccs["index"]
    for index in range(len(index_list) - 1):
        from_index = index_list[index]
        to_index = index_list[index + 1]

        x = [from_index, to_index]

        style = plot_config.getStatisticsStyle("mean")
        if style.isVisible():
            y = [ccs["mean"][from_index], ccs["mean"][to_index]]
            axes.plot(
                x,
                y,
                alpha=line_style.alpha,
                linestyle=style.line_style,
                color=line_style.color,
                linewidth=style.width,
            )

        style = plot_config.getStatisticsStyle("p50")
        if style.isVisible():
            y = [ccs["p50"][from_index], ccs["p50"][to_index]]
            axes.plot(
                x,
                y,
                alpha=line_style.alpha,
                linestyle=style.line_style,
                color=line_style.color,
                linewidth=style.width,
            )

        style = plot_config.getStatisticsStyle("std")
        if style.isVisible():
            mean = [ccs["mean"][from_index], ccs["mean"][to_index]]
            std = [ccs["std"][from_index], ccs["std"][to_index]]

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

        style = plot_config.getStatisticsStyle("min-max")
        if style.isVisible():
            y_1 = [ccs["min"][from_index], ccs["min"][to_index]]
            y_2 = [ccs["max"][from_index], ccs["max"][to_index]]

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

        style = plot_config.getStatisticsStyle("p10-p90")
        if style.isVisible():
            y_1 = [ccs["p10"][from_index], ccs["p10"][to_index]]
            y_2 = [ccs["p90"][from_index], ccs["p90"][to_index]]

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

        style = plot_config.getStatisticsStyle("p33-p67")
        if style.isVisible():
            y_1 = [ccs["p33"][from_index], ccs["p33"][to_index]]
            y_2 = [ccs["p67"][from_index], ccs["p67"][to_index]]

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
