def plotHistory(plot_context, axes):
    """
    @type axes: matplotlib.axes.Axes
    @type plot_config: PlotConfig
    """
    plot_config = plot_context.plotConfig()

    if (
        not plot_config.isHistoryEnabled()
        or plot_context.history_data is None
        or plot_context.history_data.empty
    ):
        return

    data = plot_context.history_data

    style = plot_config.historyStyle()

    lines = axes.plot_date(
        x=data.index.values,
        y=data,
        color=style.color,
        alpha=style.alpha,
        marker=style.marker,
        linestyle=style.line_style,
        linewidth=style.width,
        markersize=style.size,
    )

    if len(lines) > 0 and style.isVisible():
        plot_config.addLegendItem("History", lines[0])
