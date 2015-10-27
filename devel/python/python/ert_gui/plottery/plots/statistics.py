from matplotlib.patches import Rectangle
from pandas import DataFrame
from .refcase import plotRefcase
from .observations import plotObservations
from .plot_tools import PlotTools


def plotStatistics(plot_context):
    """
    @type plot_context: PlotContext
    """
    ert = plot_context.ert()
    key = plot_context.key()
    config = plot_context.plotConfig()
    axes = plot_context.figure().add_subplot(111)
    """:type: matplotlib.axes.Axes """

    case_list = plot_context.cases()
    for case in case_list:
        data = plot_context.dataGatherer().gatherData(ert, case, key)
        if not data.empty:
            if not data.index.is_all_dates:
                config.deactivateDateSupport()

            statistics_data = DataFrame()

            statistics_data["Minimum"] = data.min(axis=1)
            statistics_data["Maximum"] = data.max(axis=1)
            statistics_data["Mean"] = data.mean(axis=1)
            statistics_data["p10"] = data.quantile(0.1, axis=1)
            statistics_data["p33"] = data.quantile(0.33, axis=1)
            statistics_data["p50"] = data.quantile(0.50, axis=1)
            statistics_data["p67"] = data.quantile(0.67, axis=1)
            statistics_data["p90"] = data.quantile(0.90, axis=1)

            _plotPercentiles(axes, config, statistics_data, case)
            config.nextColor()

    plotRefcase(plot_context, axes)
    plotObservations(plot_context, axes)

    default_x_label = "Date" if config.isDateSupportActive() else "Index"
    PlotTools.finalizePlot(plot_context, axes, default_x_label=default_x_label, default_y_label="Value")


def _plotPercentiles(axes, plot_config, data, ensemble_label):
    """
    @type axes: matplotlib.axes.Axes
    @type plot_config: ert_gui.plottery.PlotConfig
    @type data: DataFrame
    @type ensemble_label: Str
    """
    style = plot_config.getStatisticsStyle("mean")
    if style.line_style != "":
        line = axes.plot(data.index.values, data["Mean"].values, alpha=style.alpha, linestyle=style.line_style, color=style.color, marker=style.marker)
        plot_config.addLegendItem(style.name, line[0])

    style = plot_config.getStatisticsStyle("p50")
    if style.line_style != "":
        line = axes.plot(data.index.values, data["p50"].values, alpha=style.alpha, linestyle=style.line_style, color=style.color, marker=style.marker)
        plot_config.addLegendItem(style.name, line[0])

    style = plot_config.getStatisticsStyle("min-max")
    _plotPercentile(axes, plot_config, style, data.index.values, data["Maximum"].values, data["Minimum"].values, 0.2)

    style = plot_config.getStatisticsStyle("p10-p90")
    _plotPercentile(axes, plot_config, style, data.index.values, data["p90"].values, data["p10"].values, 0.3)

    style = plot_config.getStatisticsStyle("p33-p67")
    _plotPercentile(axes, plot_config, style, data.index.values, data["p67"].values, data["p33"].values, 0.4)


def _plotPercentile(axes, plot_config, style, index_values, top_line_data, bottom_line_data, alpha_multiplier):
    alpha = style.alpha
    line_style = style.line_style
    color = style.color
    marker = style.marker

    if line_style == "#":
        axes.fill_between(index_values, bottom_line_data, top_line_data, alpha=alpha * alpha_multiplier, color=color)
        rectangle = Rectangle((0, 0), 1, 1, color=color, alpha=alpha * alpha_multiplier) # creates rectangle patch for legend use.
        plot_config.addLegendItem(style.name, rectangle)
    elif line_style != "":
        bottom_line = axes.plot(index_values, bottom_line_data, alpha=alpha, linestyle=line_style, color=color, marker=marker)
        top_line = axes.plot(index_values, top_line_data, alpha=alpha, linestyle=line_style, color=color, marker=marker)
        plot_config.addLegendItem(style.name, bottom_line[0])
