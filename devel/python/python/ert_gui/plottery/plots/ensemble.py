from .refcase import plotRefcase
from .observations import plotObservations
from .plot_tools import PlotTools

def plotEnsemble(plot_context):
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

            _plotLines(axes, config, data, case)
            config.nextColor()

    plotRefcase(plot_context, axes)
    plotObservations(plot_context, axes)

    default_x_label = "Date" if config.isDateSupportActive() else "Index"
    PlotTools.finalizePlot(plot_context, axes, default_x_label=default_x_label, default_y_label="Value")


def _plotLines(axes, plot_config, data, ensemble_label):
    """
    @type axes: matplotlib.axes.Axes
    @type plot_config: PlotConfig
    @type data: DataFrame
    @type ensemble_label: Str
    """

    style = plot_config.defaultStyle()

    if plot_config.isDateSupportActive():
        lines = axes.plot_date(x=data.index.values, y=data, color=style.color, alpha=style.alpha, marker=style.marker, linestyle=style.line_style)
    else:
        lines = axes.plot(data.index.values, data, color=style.color, alpha=style.alpha, marker=style.marker, linestyle=style.line_style)

    if len(lines) > 0:
        plot_config.addLegendItem(ensemble_label, lines[0])
