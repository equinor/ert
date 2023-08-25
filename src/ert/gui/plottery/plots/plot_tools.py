from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ert.gui.plottery import PlotContext


class PlotTools:
    @staticmethod
    def showGrid(axes, plot_context):
        config = plot_context.plotConfig()
        if config.isGridEnabled():
            axes.grid()

    @staticmethod
    def showLegend(axes, plot_context):
        config = plot_context.plotConfig()
        if config.isLegendEnabled() and len(config.legendItems()) > 0:
            axes.legend(config.legendItems(), config.legendLabels(), numpoints=1)

    @staticmethod
    def _getXAxisLimits(plot_context: "PlotContext"):
        limits = plot_context.plotConfig().limits
        axis_name = plot_context.x_axis

        if axis_name == plot_context.VALUE_AXIS:
            return limits.value_limits
        elif axis_name == plot_context.COUNT_AXIS:
            return None  # Histogram takes care of itself
        elif axis_name == plot_context.DATE_AXIS:
            return limits.date_limits
        elif axis_name == plot_context.DENSITY_AXIS:
            return limits.density_limits
        elif axis_name == plot_context.DEPTH_AXIS:
            return limits.depth_limits
        elif axis_name == plot_context.INDEX_AXIS:
            return limits.index_limits

        return None  # No limits set

    @staticmethod
    def _getYAxisLimits(plot_context: "PlotContext"):
        limits = plot_context.plotConfig().limits
        axis_name = plot_context.y_axis

        if axis_name == plot_context.VALUE_AXIS:
            return limits.value_limits
        elif axis_name == plot_context.COUNT_AXIS:
            return None  # Histogram takes care of itself
        elif axis_name == plot_context.DATE_AXIS:
            return limits.date_limits
        elif axis_name == plot_context.DENSITY_AXIS:
            return limits.density_limits
        elif axis_name == plot_context.DEPTH_AXIS:
            return limits.depth_limits
        elif axis_name == plot_context.INDEX_AXIS:
            return limits.index_limits

        return None  # No limits set

    @staticmethod
    def finalizePlot(
        plot_context: "PlotContext",
        figure,
        axes,
        default_x_label="Unnamed",
        default_y_label="Unnamed",
    ):
        PlotTools.showLegend(axes, plot_context)
        PlotTools.showGrid(axes, plot_context)

        PlotTools.__setupLabels(plot_context, default_x_label, default_y_label)

        plot_config = plot_context.plotConfig()
        axes.set_xlabel(plot_config.xLabel())
        axes.set_ylabel(plot_config.yLabel())

        x_axis_limits = PlotTools._getXAxisLimits(plot_context)
        if x_axis_limits is not None:
            axes.set_xlim(*x_axis_limits)

        y_axis_limits = PlotTools._getYAxisLimits(plot_context)
        if y_axis_limits is not None:
            axes.set_ylim(*y_axis_limits)

        axes.set_title(plot_config.title())

        if plot_context.isDateSupportActive():
            figure.autofmt_xdate()

    @staticmethod
    def __setupLabels(plot_context, default_x_label, default_y_label):
        config = plot_context.plotConfig()

        if config.xLabel() is None:
            config.setXLabel(default_x_label)

        if config.yLabel() is None:
            config.setYLabel(default_y_label)
