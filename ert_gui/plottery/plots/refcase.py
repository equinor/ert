def plotRefcase(plot_context, axes):
    plot_config = plot_context.plotConfig()

    if (
        not plot_config.isRefcaseEnabled()
        or plot_context.refcase_data is None
        or plot_context.refcase_data.empty
    ):
        return

    data = plot_context.refcase_data
    style = plot_config.refcaseStyle()

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
        plot_config.addLegendItem("Refcase", lines[0])
