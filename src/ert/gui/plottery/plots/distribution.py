import pandas as pd

from .plot_tools import PlotTools


class DistributionPlot:
    def __init__(self):
        self.dimensionality = 1

    def plot(self, figure, plot_context, case_to_data_map, _observation_data):
        plotDistribution(figure, plot_context, case_to_data_map, _observation_data)


def plotDistribution(figure, plot_context, case_to_data_map, _observation_data):
    """@type plot_context: ert.gui.plottery.PlotContext"""
    config = plot_context.plotConfig()
    axes = figure.add_subplot(111)

    plot_context.deactivateDateSupport()

    plot_context.y_axis = plot_context.VALUE_AXIS

    if plot_context.log_scale:
        axes.set_yscale("log")

    case_list = plot_context.cases()
    case_indexes = []
    previous_data = None
    for case_index, (case, data) in enumerate(case_to_data_map.items()):
        case_indexes.append(case_index)

        if not data.empty:
            _plotDistribution(axes, config, data, case, case_index, previous_data)
            config.nextColor()

        previous_data = data

    axes.set_xticks([-1] + case_indexes + [len(case_indexes)])

    rotation = 0
    if len(case_list) > 3:
        rotation = 30

    axes.set_xticklabels([""] + case_list + [""], rotation=rotation)

    config.setLegendEnabled(False)

    PlotTools.finalizePlot(
        plot_context, figure, axes, default_x_label="Case", default_y_label="Value"
    )


def _plotDistribution(axes, plot_config, data, label, index, previous_data):
    """
    @type axes: matplotlib.axes.Axes
    @type plot_config: PlotConfig
    @type data: DataFrame
    @type label: Str
    """

    if data.empty:
        data = pd.Series(dtype="float64")
    else:
        data = data[0]

    axes.set_xlabel(plot_config.xLabel())
    axes.set_ylabel(plot_config.yLabel())

    style = plot_config.distributionStyle()

    if data.dtype == "object":
        try:
            data = pd.to_numeric(data, errors="coerce")
        except AttributeError:
            data = data.convert_objects(convert_numeric=True)

    if data.dtype == "object":
        dots = []
    else:
        dots = axes.plot(
            [index] * len(data),
            data,
            color=style.color,
            alpha=style.alpha,
            marker=style.marker,
            linestyle=style.line_style,
            markersize=style.size,
        )

        if plot_config.isDistributionLineEnabled() and previous_data is not None:
            line_style = plot_config.distributionLineStyle()
            x = [index - 1, index]
            y = [previous_data[0], data]
            axes.plot(
                x,
                y,
                color=line_style.color,
                alpha=line_style.alpha,
                linestyle=line_style.line_style,
                linewidth=line_style.width,
            )

    if len(dots) > 0:
        plot_config.addLegendItem(label, dots[0])
