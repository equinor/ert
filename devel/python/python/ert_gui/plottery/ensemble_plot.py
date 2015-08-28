from .plot_context import PlotConfig
from matplotlib.patches import Rectangle

class EnsemblePlot(object):

    @staticmethod
    def plotLines(axes, plot_config, data, ensemble_label):
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


    @staticmethod
    def plotRefcase(axes, plot_config, data):
        """
        @type axes: matplotlib.axes.Axes
        @type plot_config: PlotConfig
        @type data: DataFrame
        """

        line_color = plot_config.refcaseColor()
        line_alpha = plot_config.refcaseAlpha()
        line_marker = plot_config.refcaseMarker()
        line_style = plot_config.refcaseStyle()
        line_width = plot_config.refcaseWidth()

        lines = axes.plot_date(x=data.index.values, y=data, color=line_color, alpha=line_alpha, marker=line_marker, linestyle=line_style, linewidth=line_width)

        if len(lines) > 0:
            plot_config.addLegendItem("Refcase", lines[0])


    @staticmethod
    def plotArea(axes, plot_config, data, ensemble_label):
        """
        @type axes: matplotlib.axes.Axes
        @type plot_config: PlotConfig
        @type data: DataFrame
        @type ensemble_label: Str
        """

        axes.set_xlabel(plot_config.xLabel())
        axes.set_ylabel(plot_config.yLabel())

        line_color = plot_config.lineColor()
        line_alpha = plot_config.lineAlpha() * 0.5

        poly_collection = axes.fill_between(data.index.values, data["Minimum"].values, data["Maximum"].values, alpha=line_alpha, color=line_color)

        rectangle = Rectangle((0, 0), 1, 1, color=line_color) # creates rectangle patch for legend use.

        plot_config.addLegendItem(ensemble_label, rectangle)

    @staticmethod
    def plotPercentiles(axes, plot_config, data, ensemble_label):
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

        minimum_line = axes.plot(data.index.values, data["Minimum"].values, alpha=line_alpha, linestyle="--", color=line_color)
        maximum_line = axes.plot(data.index.values, data["Maximum"].values, alpha=line_alpha, linestyle="--", color=line_color)
        p50_line = axes.plot(data.index.values, data["p50"].values, alpha=line_alpha, linestyle="--", color=line_color, marker="x")
        mean_line = axes.plot(data.index.values, data["Mean"].values, alpha=line_alpha, linestyle="-", color=line_color, marker="")
        axes.fill_between(data.index.values, data["p10"].values, data["p90"].values, alpha=line_alpha * 0.3, color=line_color)
        axes.fill_between(data.index.values, data["p33"].values, data["p67"].values, alpha=line_alpha * 0.5, color=line_color)

        rectangle_p10_p90 = Rectangle((0, 0), 1, 1, color=line_color, alpha=line_alpha * 0.3) # creates rectangle patch for legend use.
        rectangle_p33_p67 = Rectangle((0, 0), 1, 1, color=line_color, alpha=line_alpha * 0.5) # creates rectangle patch for legend use.

        plot_config.addLegendItem("Minimum/Maximum", minimum_line[0])
        plot_config.addLegendItem("P50", p50_line[0])
        plot_config.addLegendItem("Mean", mean_line[0])
        plot_config.addLegendItem("P10-P90", rectangle_p10_p90)
        plot_config.addLegendItem("P33-P67", rectangle_p33_p67)

        # plot_config.addLegendItem("%s Minimum/Maximum" % ensemble_label, minimum_line[0])
        # plot_config.addLegendItem("%s P50" % ensemble_label, p50_line[0])
        # plot_config.addLegendItem("%s Mean" % ensemble_label, mean_line[0])
        # plot_config.addLegendItem("%s P10 - P90" % ensemble_label, rectangle_p10_p90)
        # plot_config.addLegendItem("%s P33 - P67" % ensemble_label, rectangle_p33_p67)
