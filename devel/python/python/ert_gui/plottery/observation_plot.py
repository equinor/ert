from .plot_config import PlotConfig
class ObservationPlot(object):

    @staticmethod
    def plot(axes, plot_config, data, value_column):
        """
        @type axes: matplotlib.axes.Axes
        @type plot_config: PlotConfig
        @type data: DataFrame
        @type value_column: Str
        """

        error_color = plot_config.observationsColor()
        error_alpha = plot_config.observationsAlpha()

        data = data.dropna()
        errorbars = axes.errorbar(x=data.index.values, y=data[value_column], yerr=data["STD_%s" % value_column],
                     fmt='none', ecolor=error_color, alpha=error_alpha)


        # if len(lines) > 0:
        #     plot_config.addLegendItem(ensemble_label, lines[0])