from math import ceil, floor, log10, sqrt
from typing import TYPE_CHECKING, List

import numpy
import pandas as pd
from matplotlib.patches import Rectangle

from .plot_tools import PlotTools

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ert.gui.plottery import PlotConfig, PlotContext


class HistogramPlot:
    def __init__(self):
        self.dimensionality = 1

    def plot(self, figure, plot_context, ensemble_to_data_map, _observation_data):
        plotHistogram(figure, plot_context, ensemble_to_data_map, _observation_data)


def plotHistogram(
    figure, plot_context: "PlotContext", ensemble_to_data_map, _observation_data
):
    config = plot_context.plotConfig()

    ensemble_list = plot_context.ensembles()
    if not ensemble_list:
        dummy_ensemble_name = "default"
        ensemble_list = [dummy_ensemble_name]
        ensemble_to_data_map = {dummy_ensemble_name: pd.DataFrame()}

    ensemble_count = len(ensemble_list)

    plot_context.x_axis = plot_context.VALUE_AXIS
    plot_context.Y_axis = plot_context.COUNT_AXIS

    if config.xLabel() is None:
        config.setXLabel("Value")

    if config.yLabel() is None:
        config.setYLabel("Count")

    use_log_scale = plot_context.log_scale

    data = {}
    minimum = None
    maximum = None
    categories = set()
    max_element_count = 0
    categorical = False
    for ensemble, datas in ensemble_to_data_map.items():
        if datas.empty:
            data[ensemble] = pd.Series(dtype="float64")
            continue

        data[ensemble] = datas[0]

        if data[ensemble].dtype == "object":
            try:
                data[ensemble] = pd.to_numeric(data[ensemble], errors="ignore")
            except AttributeError:
                data[ensemble] = data[ensemble].convert_objects(convert_numeric=True)

        if data[ensemble].dtype == "object":
            categorical = True

        if categorical:
            categories = categories.union(set(data[ensemble].unique()))
        else:
            current_min = data[ensemble].min()
            current_max = data[ensemble].max()
            minimum = current_min if minimum is None else min(minimum, current_min)

            maximum = current_max if maximum is None else max(maximum, current_max)

            max_element_count = max(max_element_count, len(data[ensemble].index))

    categories = sorted(categories)
    bin_count = int(ceil(sqrt(max_element_count)))

    axes = {}
    for index, ensemble in enumerate(ensemble_list):
        axes[ensemble] = figure.add_subplot(ensemble_count, 1, index + 1)

        axes[ensemble].set_title(f"{config.title()} ({ensemble})")

        if use_log_scale:
            axes[ensemble].set_xscale("log")

        if not data[ensemble].empty:
            if categorical:
                _plotCategoricalHistogram(
                    axes[ensemble], config, data[ensemble], ensemble, categories
                )
            else:
                _plotHistogram(
                    axes[ensemble],
                    config,
                    data[ensemble],
                    ensemble,
                    bin_count,
                    use_log_scale,
                    minimum,
                    maximum,
                )

            config.nextColor()
            PlotTools.showGrid(axes[ensemble], plot_context)

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
    plot_config: "PlotConfig",
    data: pd.DataFrame,
    label: str,
    categories: List[str],
):
    axes.set_xlabel(plot_config.xLabel())
    axes.set_ylabel(plot_config.yLabel())

    style = plot_config.histogramStyle()

    counts = data.value_counts()
    freq = [counts.get(category, 0) for category in categories]
    pos = numpy.arange(len(categories))
    width = 1.0
    axes.set_xticks(pos + (width / 2.0))
    axes.set_xticklabels(categories)

    axes.bar(pos, freq, alpha=style.alpha, color=style.color, width=width)

    rectangle = Rectangle(
        (0, 0), 1, 1, color=style.color
    )  # creates rectangle patch for legend use.
    plot_config.addLegendItem(label, rectangle)


def _plotHistogram(
    axes: "Axes",
    plot_config: "PlotConfig",
    data: pd.DataFrame,
    label: str,
    bin_count,
    use_log_scale=False,
    minimum=None,
    maximum=None,
):
    axes.set_xlabel(plot_config.xLabel())
    axes.set_ylabel(plot_config.yLabel())

    style = plot_config.histogramStyle()

    if minimum is not None and maximum is not None:
        if use_log_scale:
            bins = _histogramLogBins(bin_count, minimum, maximum)
        else:
            bins = numpy.linspace(minimum, maximum, bin_count)
    else:
        bins = bin_count

    axes.hist(data.values, alpha=style.alpha, bins=bins, color=style.color)

    if minimum == maximum:
        minimum -= 0.5
        maximum += 0.5

    axes.set_xlim(minimum, maximum)

    rectangle = Rectangle(
        (0, 0), 1, 1, color=style.color
    )  # creates rectangle patch for legend use.'
    plot_config.addLegendItem(label, rectangle)


def _histogramLogBins(bin_count, minimum=None, maximum=None):
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
