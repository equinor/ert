from __future__ import annotations

from collections.abc import Sequence
from math import ceil, floor, log10, sqrt
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from ert.gui.plotting.plot_api import EnsembleObject, PlotApiKeyDefinition
from ert.gui.plotting.utils.plot_tools import ConditionalAxisFormatter, PlotTools
from ert.shared.status.utils import convert_to_numeric

if TYPE_CHECKING:
    import numpy.typing as npt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from ert.gui.plotting.utils import PlotContext, PlotStyle
    from ert.gui.plotting.utils.plot_types import ObservationPlotLocations


class HistogramPlot:
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
    plot_context.deactivate_date_support()

    x_label = config.x_label()
    if x_label is None:
        x_label = "Value"
    config.set_x_label(x_label)

    y_label = config.y_label()
    if y_label is None:
        y_label = "Count"
    config.set_y_label(y_label)

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

        if not pd.api.types.is_numeric_dtype(data[ensemble.id]):
            data[ensemble.id] = convert_to_numeric(data[ensemble.id])

        if not pd.api.types.is_numeric_dtype(data[ensemble.id]):
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
    for (index, ensemble), color_index in zip(
        enumerate(ensemble_list), plot_context.ensembles_color_indexes(), strict=False
    ):
        config.set_current_color(color_index)
        axes[ensemble.name] = figure.add_subplot(ensemble_count, 1, index + 1)

        axes[ensemble.name].set_title(
            f"{config.title()} ({ensemble.experiment_name} : {ensemble.name})"
        )
        axes[ensemble.name].set_xlabel(x_label)
        axes[ensemble.name].set_ylabel(y_label)
        is_last = index == ensemble_count - 1

        if ensemble.id in data and not data[ensemble.id].empty:
            if categorical:
                config.add_legend_item(
                    ensemble.name,
                    _plotCategoricalHistogram(
                        axes[ensemble.name],
                        config.histogram_style(),
                        data[ensemble.id],
                        sorted(categories),
                    ),
                )
            else:
                if minimum is not None and maximum is not None and minimum == maximum:
                    minimum -= 0.1
                    maximum += 0.1

                legend_patch = _plotHistogram(
                    axes[ensemble.name],
                    config.histogram_style(),
                    data[ensemble.id],
                    bin_count,
                    use_log_scale=plot_context.log_scale,
                    minimum=minimum,
                    maximum=maximum,
                )
                axes[ensemble.name].legend(
                    [legend_patch],
                    [f"{ensemble.experiment_name} : {ensemble.name}"],
                    numpoints=1,
                )
                if index != 0:
                    plot_context.plotConfig().set_title("")
                PlotTools.finalizePlot(
                    plot_context,
                    figure,
                    axes[ensemble.name],
                    default_x_label=axes[ensemble.name].get_xlabel(),
                    default_y_label=axes[ensemble.name].get_ylabel(),
                )
                # Removes x-axis labels for all but the last subplot to avoid clutter
                if not is_last:
                    axes[ensemble.name].set_xlabel("")

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
    width = 0.95
    axes.set_xticks(pos)
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
    *,
    use_log_scale: bool = False,
    minimum: float | None = None,
    maximum: float | None = None,
) -> Rectangle:
    bins: Sequence[float] | int
    if minimum is not None and maximum is not None:
        # Ensure we have at least 2 bin edges to create 1 bin
        effective_bin_count = max(bin_count + 1, 2)
        if use_log_scale:
            bins = _histogramLogBins(effective_bin_count, minimum, maximum)  # type: ignore
            axes.set_xscale("log")
        else:
            bins = np.linspace(minimum, maximum, effective_bin_count)  # type: ignore
            axes.set_xscale("linear")
            axes.xaxis.set_major_formatter(ConditionalAxisFormatter())

    else:
        bins = bin_count

    axes.hist(data.values, alpha=style.alpha, bins=bins, color=style.color)

    axes.set_xlim(minimum, maximum)

    return Rectangle(
        (0, 0), 1, 1, color=style.color
    )  # creates rectangle patch for legend use.'


def _histogramLogBins(
    bin_count: int, minimum: float, maximum: float
) -> npt.NDArray[np.floating[Any]]:
    minimum = log10(float(minimum))
    maximum = log10(float(maximum))

    min_value = floor(minimum)
    max_value = ceil(maximum)

    log_bin_count = max_value - min_value

    if log_bin_count < bin_count:
        next_bin_count = log_bin_count * 2

        if bin_count - log_bin_count > next_bin_count - bin_count:
            log_bin_count = next_bin_count
        else:
            log_bin_count = bin_count

    return 10 ** np.linspace(minimum, maximum, log_bin_count)
