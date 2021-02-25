from .observations import plotObservations
from .plot_tools import PlotTools
from ert_gui.plottery.plots.history import plotHistory
from ert_gui.plottery.plots.refcase import plotRefcase


class EnsemblePlot(object):
    def __init__(self):
        self.dimensionality = 2

    def plot(self, figure, plot_context, case_to_data_map, observation_data):
        """
        @type plot_context: ert_gui.plottery.PlotContext
        """
        config = plot_context.plotConfig()
        """:type: ert_gui.plottery.PlotConfig """
        axes = figure.add_subplot(111)
        """:type: matplotlib.axes.Axes """

        plot_context.y_axis = plot_context.VALUE_AXIS
        plot_context.x_axis = plot_context.DATE_AXIS

        for case, data in case_to_data_map.items():
            data = data.T

            if not data.empty:
                if not data.index.is_all_dates:
                    plot_context.deactivateDateSupport()
                    plot_context.x_axis = plot_context.INDEX_AXIS

                self._plotLines(
                    axes, config, data, case, plot_context.isDateSupportActive()
                )
                config.nextColor()

        plotRefcase(plot_context, axes)
        plotObservations(observation_data, plot_context, axes)
        plotHistory(plot_context, axes)

        default_x_label = "Date" if plot_context.isDateSupportActive() else "Index"
        PlotTools.finalizePlot(
            plot_context,
            figure,
            axes,
            default_x_label=default_x_label,
            default_y_label="Value",
        )

    def _plotLines(self, axes, plot_config, data, ensemble_label, is_date_supported):
        """
        @type axes: matplotlib.axes.Axes
        @type plot_config: ert_gui.plottery.PlotConfig
        @type data: pandas.DataFrame
        @type ensemble_label: Str
        """

        style = plot_config.defaultStyle()

        if len(data) == 1 and style.marker == "":
            style.marker = "."

        if is_date_supported:
            lines = axes.plot_date(
                x=data.index.to_numpy(),
                y=data.to_numpy(),
                color=style.color,
                alpha=style.alpha,
                marker=style.marker,
                linestyle=style.line_style,
                linewidth=style.width,
                markersize=style.size,
            )
        else:
            lines = axes.plot(
                data.index.to_numpy(),
                data.to_numpy(),
                color=style.color,
                alpha=style.alpha,
                marker=style.marker,
                linestyle=style.line_style,
                linewidth=style.width,
                markersize=style.size,
            )

        if len(lines) > 0:
            plot_config.addLegendItem(ensemble_label, lines[0])
