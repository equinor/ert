from typing import TYPE_CHECKING

import numpy
from scipy.stats import gaussian_kde

from .plot_tools import PlotTools

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from pandas import DataFrame

    from ert.gui.plottery import PlotConfig, PlotContext


class GaussianKDEPlot:
    def __init__(self):
        self.dimensionality = 1

    def plot(self, figure, plot_context, case_to_data_map, _observation_data):
        plotGaussianKDE(figure, plot_context, case_to_data_map, _observation_data)


def plotGaussianKDE(
    figure, plot_context: "PlotContext", case_to_data_map, _observation_data
):
    config = plot_context.plotConfig()
    axes = figure.add_subplot(111)

    plot_context.deactivateDateSupport()
    plot_context.x_axis = plot_context.VALUE_AXIS
    plot_context.y_axis = plot_context.DENSITY_AXIS

    if plot_context.log_scale:
        axes.set_xscale("log")

    for case, data in case_to_data_map.items():
        if data.empty:
            continue
        data = data[0]
        if data.nunique() > 1:
            _plotGaussianKDE(axes, config, data, case)
            config.nextColor()

    PlotTools.finalizePlot(
        plot_context, figure, axes, default_x_label="Value", default_y_label="Density"
    )


def _plotGaussianKDE(
    axes: "Axes", plot_config: "PlotConfig", data: "DataFrame", label: str
):
    style = plot_config.histogramStyle()

    sample_range = data.max() - data.min()
    indexes = numpy.linspace(
        data.min() - 0.5 * sample_range, data.max() + 0.5 * sample_range, 1000
    )
    gkde = gaussian_kde(data.values)
    evaluated_gkde = gkde.evaluate(indexes)

    lines = axes.plot(
        indexes,
        evaluated_gkde,
        linewidth=style.width,
        color=style.color,
        alpha=style.alpha,
    )

    if len(lines) > 0:
        plot_config.addLegendItem(label, lines[0])
