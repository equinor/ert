from .plot_tools import PlotTools


def plotDistribution(plot_context):
    """
    @type plot_context: PlotContext
    """
    ert = plot_context.ert()
    key = plot_context.key()
    config = plot_context.plotConfig()
    axes = plot_context.figure().add_subplot(111)
    """:type: matplotlib.axes.Axes """

    config.deactivateDateSupport()

    if key.startswith("LOG10_"):
        key = key[6:]
        axes.set_yscale("log")

    case_list = plot_context.cases()
    case_indexes = []
    for case_index, case in enumerate(case_list):
        case_indexes.append(case_index)
        data = plot_context.dataGatherer().gatherData(ert, case, key)

        if not data.empty and data.nunique() > 1:
            _plotDistribution(axes, config, data, case, case_index)
            config.nextColor()

    axes.set_xticks([-1] + case_indexes + [len(case_indexes)])
    axes.set_xticklabels([""] + case_list + [""])

    PlotTools.finalizePlot(plot_context, axes, default_x_label="Case", default_y_label="Value")


def _plotDistribution(axes, plot_config, data, label, index):
    """
    @type axes: matplotlib.axes.Axes
    @type plot_config: PlotConfig
    @type data: DataFrame
    @type label: Str
    """

    axes.set_xlabel(plot_config.xLabel())
    axes.set_ylabel(plot_config.yLabel())

    style = plot_config.distributionStyle()

    if data.dtype == "object":
        data = data.convert_objects(convert_numeric=True)

    if data.dtype == "object":
        lines = []
    else:
        lines = axes.plot([index] * len(data), data, color=style.color, alpha=style.alpha, marker=style.marker, linestyle=style.line_style, markersize=style.width)

    if len(lines) > 0:
        plot_config.addLegendItem(label, lines[0])
