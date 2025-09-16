from __future__ import annotations

from collections.abc import Sequence
from math import ceil, sqrt
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from ert.gui.tools.plot.plot_api import EnsembleObject
from ert.shared.status.utils import convert_to_numeric

from .plot_tools import PlotTools

if TYPE_CHECKING:
    import numpy.typing as npt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from ert.gui.tools.plot.plottery import PlotContext, PlotStyle


class HistogramPlot:
    def __init__(self) -> None:
        self.dimensionality = 1

    @staticmethod
    def plot(
        figure: Figure,
        plot_context: PlotContext,
        ensemble_to_data_map: dict[EnsembleObject, pd.DataFrame],
        observation_data: pd.DataFrame,
        std_dev_images: dict[str, npt.NDArray[np.float32]],
    ) -> None:
        plotHistogram(figure, plot_context, ensemble_to_data_map)


def plotHistogram(
    figure: Figure,
    plot_context: PlotContext,
    ensemble_to_data_map: dict[EnsembleObject, pd.DataFrame],
) -> None:
    config = plot_context.plotConfig()

    ensemble_list = plot_context.ensembles()

    ensemble_count = len(ensemble_list)

    plot_context.x_axis = plot_context.VALUE_AXIS
    plot_context.y_axis = plot_context.COUNT_AXIS

    x_label = config.xLabel()
    if x_label is None:
        x_label = "Value"
    config.setXLabel(x_label)

    y_label = config.yLabel()
    if y_label is None:
        y_label = "Count"
    config.setYLabel(y_label)

    data = {}
    minimum = None
    maximum = None
    categories: set[str] = set()
    max_element_count = 0
    categorical = False

    for ensemble, datas in ensemble_to_data_map.items():
        if datas.empty:
            data[ensemble.id] = pd.Series(dtype="float64")
            continue

        data[ensemble.id] = datas[0]

        if data[ensemble.id].dtype == "object":
            try:
                data[ensemble.id] = convert_to_numeric(data[ensemble.id])
            except AttributeError:
                data[ensemble.id] = data[ensemble.id].convert_objects(
                    convert_numeric=True
                )

        if data[ensemble.id].dtype == "object":
            categorical = True

        if categorical:
            categories = categories.union(set(data[ensemble.id].unique()))
        else:
            current_min = data[ensemble.id].min()
            current_max = data[ensemble.id].max()
            minimum = current_min if minimum is None else min(minimum, current_min)
            maximum = current_max if maximum is None else max(maximum, current_max)
            max_element_count = max(max_element_count, len(data[ensemble.id].index))

    bin_count = ceil(sqrt(max_element_count))

    axes = {}
    for index, ensemble in enumerate(ensemble_list):
        axes[ensemble.name] = figure.add_subplot(ensemble_count, 1, index + 1)

        axes[ensemble.name].set_title(
            f"{config.title()} ({ensemble.experiment_name} : {ensemble.name})"
        )
        axes[ensemble.name].set_xlabel(x_label)
        axes[ensemble.name].set_ylabel(y_label)

        if ensemble.id in data and not data[ensemble.id].empty:
            if categorical:
                config.addLegendItem(
                    ensemble.name,
                    _plotCategoricalHistogram(
                        axes[ensemble.name],
                        config.histogramStyle(),
                        data[ensemble.id],
                        sorted(categories),
                    ),
                )
            else:
                if minimum is not None and maximum is not None and minimum == maximum:
                    minimum -= 0.1
                    maximum += 0.1
                config.addLegendItem(
                    ensemble.name,
                    _plotHistogram(
                        axes[ensemble.name],
                        config.histogramStyle(),
                        data[ensemble.id],
                        bin_count,
                        minimum,
                        maximum,
                    ),
                )

            config.nextColor()
            PlotTools.showGrid(axes[ensemble.name], plot_context)

    min_count = 0
    max_count = (
        max(subplot.get_ylim()[1] for subplot in axes.values()) if axes.values() else 0
    )

    custom_limits = plot_context.plotConfig().limits

    if custom_limits.count_maximum is not None:
        max_count = custom_limits.count_maximum

    if custom_limits.count_minimum is not None:
        min_count = custom_limits.count_minimum

    for subplot in axes.values():
        subplot.set_ylim(min_count, max_count)
        subplot.set_xlim(custom_limits.value_minimum, custom_limits.value_maximum)


def _plotCategoricalHistogram(
    axes: Axes,
    style: PlotStyle,
    data: pd.DataFrame,
    categories: list[str],
) -> Rectangle:
    counts = data.value_counts()
    freq = [counts.get(category, 0) for category in categories]
    pos = np.arange(len(categories))
    width = 1.0
    axes.set_xticks(pos + (width / 2.0))
    axes.set_xticklabels(categories)

    axes.bar(pos, freq, alpha=style.alpha, color=style.color, width=width)

    return Rectangle(
        (0, 0), 1, 1, color=style.color
    )  # creates rectangle patch for legend use.


def _plotHistogram(
    axes: Axes,
    style: PlotStyle,
    data: pd.DataFrame,
    bin_count: int,
    minimum: float | None = None,
    maximum: float | None = None,
) -> Rectangle:
    bins: Sequence[float] | int
    if minimum is not None and maximum is not None:
        # Ensure we have at least 2 bin edges to create 1 bin
        effective_bin_count = max(bin_count + 1, 2)
        bins = np.linspace(minimum, maximum, effective_bin_count)  # type: ignore

        if minimum == maximum:
            minimum -= 0.5
            maximum += 0.5
    else:
        bins = bin_count

    axes.hist(data.values, alpha=style.alpha, bins=bins, color=style.color)

    axes.set_xlim(minimum, maximum)

    return Rectangle(
        (0, 0), 1, 1, color=style.color
    )  # creates rectangle patch for legend use.'
