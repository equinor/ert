from __future__ import annotations

from math import ceil, floor, log10, sqrt
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union

import numpy
import pandas as pd
from matplotlib.patches import Rectangle

from ert.gui.tools.plot.plot_api import EnsembleObject

from .plot_tools import PlotTools

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

    from ert.gui.plottery import PlotContext, PlotStyle


class HistogramPlot:
    def __init__(self) -> None:
        self.dimensionality = 1

    @staticmethod
    def plot(
        figure: Figure,
        plot_context: PlotContext,
        ensemble_to_data_map: Dict[EnsembleObject, pd.DataFrame],
        observation_data: pd.DataFrame,
        std_dev_images: Dict[str, bytes],
    ) -> None:
        plotHistogram(figure, plot_context, ensemble_to_data_map, observation_data)


def plotHistogram(
    figure: Figure,
    plot_context: PlotContext,
    ensemble_to_data_map: Dict[EnsembleObject, pd.DataFrame],
    observation_data: pd.DataFrame,
) -> None:
    config = plot_context.plotConfig()

    ensemble_list = plot_context.ensembles()
    if not ensemble_list:
        dummy_ensemble_name = "default"
        dummy_ensemble_object = EnsembleObject(
            name=dummy_ensemble_name,
            hidden=False,
            id="id",
            experiment_name="default",
        )
        ensemble_list = [dummy_ensemble_object]
        ensemble_to_data_map = {dummy_ensemble_object: pd.DataFrame()}

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

    use_log_scale = plot_context.log_scale

    data = {}
    minimum = None
    maximum = None
    categories: set[str] = set()
    max_element_count = 0
    categorical = False
    for ensemble, datas in ensemble_to_data_map.items():
        if datas.empty:
            data[ensemble.name] = pd.Series(dtype="float64")
            continue

        data[ensemble.name] = datas[0]

        if data[ensemble.name].dtype == "object":
            try:
                data[ensemble.name] = pd.to_numeric(
                    data[ensemble.name], errors="ignore"
                )
            except AttributeError:
                data[ensemble.name] = data[ensemble.name].convert_objects(
                    convert_numeric=True
                )

        if data[ensemble.name].dtype == "object":
            categorical = True

        if categorical:
            categories = categories.union(set(data[ensemble.name].unique()))
        else:
            current_min = data[ensemble.name].min()
            current_max = data[ensemble.name].max()
            minimum = current_min if minimum is None else min(minimum, current_min)  # type: ignore
            maximum = current_max if maximum is None else max(maximum, current_max)  # type: ignore
            max_element_count = max(max_element_count, len(data[ensemble.name].index))

    bin_count = int(ceil(sqrt(max_element_count)))

    axes = {}
    for index, ensemble in enumerate(ensemble_list):
        axes[ensemble.name] = figure.add_subplot(ensemble_count, 1, index + 1)

        axes[ensemble.name].set_title(
            f"{config.title()} ({ensemble.experiment_name} : {ensemble.name})"
        )
        axes[ensemble.name].set_xlabel(x_label)
        axes[ensemble.name].set_ylabel(y_label)

        if use_log_scale:
            axes[ensemble.name].set_xscale("log")

        if not data[ensemble.name].empty:
            if categorical:
                config.addLegendItem(
                    ensemble.name,
                    _plotCategoricalHistogram(
                        axes[ensemble.name],
                        config.histogramStyle(),
                        data[ensemble.name],
                        sorted(categories),
                    ),
                )
            else:
                config.addLegendItem(
                    ensemble.name,
                    _plotHistogram(
                        axes[ensemble.name],
                        config.histogramStyle(),
                        data[ensemble.name],
                        bin_count,
                        use_log_scale,
                        minimum,
                        maximum,
                    ),
                )

            config.nextColor()
            PlotTools.showGrid(axes[ensemble.name], plot_context)

    min_count = 0
    max_count = max(subplot.get_ylim()[1] for subplot in axes.values())

    custom_limits = plot_context.plotConfig().limits

    if custom_limits.count_maximum is not None:
        max_count = custom_limits.count_maximum

    if custom_limits.count_minimum is not None:
        min_count = custom_limits.count_minimum

    for subplot in axes.values():
        subplot.set_ylim(min_count, max_count)
        subplot.set_xlim(custom_limits.value_minimum, custom_limits.value_maximum)


def _plotCategoricalHistogram(
    axes: "Axes",
    style: PlotStyle,
    data: pd.DataFrame,
    categories: List[str],
) -> Rectangle:
    counts = data.value_counts()
    freq = [counts.get(category, 0) for category in categories]
    pos = numpy.arange(len(categories))
    width = 1.0
    axes.set_xticks(pos + (width / 2.0))
    axes.set_xticklabels(categories)

    axes.bar(pos, freq, alpha=style.alpha, color=style.color, width=width)

    return Rectangle(
        (0, 0), 1, 1, color=style.color
    )  # creates rectangle patch for legend use.


def _plotHistogram(
    axes: "Axes",
    style: PlotStyle,
    data: pd.DataFrame,
    bin_count: int,
    use_log_scale: float = False,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> Rectangle:
    bins: Union[Sequence[float], int]
    if minimum is not None and maximum is not None:
        if use_log_scale:
            bins = _histogramLogBins(bin_count, minimum, maximum)  # type: ignore
        else:
            bins = numpy.linspace(minimum, maximum, bin_count)  # type: ignore

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


def _histogramLogBins(
    bin_count: int, minimum: float, maximum: float
) -> NDArray[numpy.floating[Any]]:
    minimum = log10(float(minimum))
    maximum = log10(float(maximum))

    min_value = int(floor(minimum))
    max_value = int(ceil(maximum))

    log_bin_count = max_value - min_value

    if log_bin_count < bin_count:
        next_bin_count = log_bin_count * 2

        if bin_count - log_bin_count > next_bin_count - bin_count:
            log_bin_count = next_bin_count
        else:
            log_bin_count = bin_count

    return 10 ** numpy.linspace(minimum, maximum, log_bin_count)
