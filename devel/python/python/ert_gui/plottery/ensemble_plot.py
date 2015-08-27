from .plot_context import PlotConfig

class EnsemblePlot(object):

    @staticmethod
    def plot(axes, plot_config, data, ensemble_label):
        """
        @type axes: matplotlib.axes.Axes
        @type plot_config: PlotConfig
        @type data: DataFrame
        @type ensemble_label: Str
        """

        axes.set_xlabel(plot_config.xLabel())
        axes.set_ylabel(plot_config.yLabel())

        line_color = plot_config.lineColor()
        line_alpha = plot_config.lineAlpha()
        line_marker = plot_config.lineMarker()
        line_style = plot_config.lineStyle()
        lines = axes.plot_date(x=data.index.values, y=data, color=line_color, alpha=line_alpha, marker=line_marker, linestyle=line_style)

        if len(lines) > 0:
            plot_config.addLegendItem(ensemble_label, lines[0])

