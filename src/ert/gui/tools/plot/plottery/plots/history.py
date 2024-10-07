from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ert.gui.tools.plot.plottery import PlotContext


def plotHistory(plot_context: "PlotContext", axes: "Axes") -> None:
    plot_config = plot_context.plotConfig()

    if (
        not plot_config.isHistoryEnabled()
        or plot_context.history_data is None
        or plot_context.history_data.empty
    ):
        return

    style = plot_config.historyStyle()

    lines = axes.plot(
        plot_context.history_data.index.values,
        plot_context.history_data,
        color=style.color,
        alpha=style.alpha,
        linewidth=style.width,
        markersize=style.size,
        marker=style.marker,
        linestyle=style.line_style,
    )

    if len(lines) > 0 and style.isVisible():
        plot_config.addLegendItem("History", lines[0])
