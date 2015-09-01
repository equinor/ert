from math import ceil, sqrt, floor, log10
from matplotlib.patches import Rectangle
import numpy
from scipy.stats import gaussian_kde


class ProbabilityPlot(object):

    @staticmethod
    def plotGaussianKDE(axes, plot_config, data, label):
        """
        @type axes: matplotlib.axes.Axes
        @type plot_config: PlotConfig
        @type data: DataFrame
        @type label: Str
        """

        axes.set_xlabel(plot_config.xLabel())
        axes.set_ylabel(plot_config.yLabel())

        line_color = plot_config.lineColor()
        line_alpha = plot_config.lineAlpha()
        line_marker = plot_config.lineMarker()
        line_style = plot_config.lineStyle()
        line_width = 2

        sample_range = data.max() - data.min()
        indexes = numpy.linspace(data.min() - 0.5 * sample_range, data.max() + 0.5 * sample_range, 1000)
        gkde = gaussian_kde(data)
        evaluated_gkde = gkde.evaluate(indexes)

        lines = axes.plot(indexes, evaluated_gkde, linewidth=line_width, color=line_color, alpha=line_alpha)

        if len(lines) > 0:
            plot_config.addLegendItem(label, lines[0])

    @staticmethod
    def plotHistogram(axes, plot_config, data, label, use_log_scale=False, minimum=None, maximum=None):
        """
        @type axes: matplotlib.axes.Axes
        @type plot_config: PlotConfig
        @type data: DataFrame
        @type label: Str
        """

        axes.set_xlabel(plot_config.xLabel())
        axes.set_ylabel(plot_config.yLabel())

        line_color = plot_config.lineColor()
        line_alpha = plot_config.lineAlpha()
        line_marker = plot_config.lineMarker()
        line_style = plot_config.lineStyle()
        line_width = 2

        bins = int(ceil(sqrt(len(data.index))))

        if use_log_scale:
            bins = ProbabilityPlot._histogramLogBins(data, bins, minimum, maximum)
        elif minimum is not None and maximum is not None:
            bins = numpy.linspace(minimum, maximum, bins)

        axes.hist(data.values, alpha=line_alpha, bins=bins, color=line_color)

        rectangle = Rectangle((0, 0), 1, 1, color=line_color) # creates rectangle patch for legend use.'
        plot_config.addLegendItem(label, rectangle)



    @staticmethod
    def _histogramLogBins(data, bin_count, minimum=None, maximum=None):
        """
        @type data: pandas.DataFrame
        @rtype: int
        """

        if minimum is None:
            minimum = data.min()

        if maximum is None:
            maximum = data.max()

        minimum = log10(float(minimum))
        maximum = log10(float(maximum))

        min_value = int(floor(minimum))
        max_value = int(ceil(maximum))

        log_bin_count = max_value - min_value

        if log_bin_count < bin_count:
            next_bin_count = log_bin_count * 2

            if bin_count - log_bin_count > next_bin_count - bin_count:
                log_bin_count = next_bin_count
            else:
                log_bin_count = bin_count

        return 10 ** numpy.linspace(minimum, maximum, log_bin_count)