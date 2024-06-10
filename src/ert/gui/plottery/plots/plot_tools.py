from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from datetime import date

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from ert.gui.plottery import PlotContext


class PlotTools:
    @staticmethod
    def showGrid(axes: Axes, plot_context: PlotContext) -> None:
        config = plot_context.plotConfig()
        if config.isGridEnabled():
            axes.grid()

    @staticmethod
    def showLegend(axes: Axes, plot_context: PlotContext) -> None:
        config = plot_context.plotConfig()
        if config.isLegendEnabled() and len(config.legendItems()) > 0:
            axes.legend(config.legendItems(), config.legendLabels(), numpoints=1)

    @staticmethod
    def _getXAxisLimits(
        plot_context: PlotContext,
    ) -> Optional[
        tuple[Optional[int], Optional[int]]
        | tuple[Optional[float], Optional[float]]
        | tuple[Optional[date], Optional[date]]
    ]:
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
        elif axis_name == plot_context.INDEX_AXIS:
            return limits.index_limits

        return None  # No limits set

    @staticmethod
    def _getYAxisLimits(
        plot_context: PlotContext,
    ) -> Optional[
        tuple[Optional[int], Optional[int]]
        | tuple[Optional[float], Optional[float]]
        | tuple[Optional[date], Optional[date]]
    ]:
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
        elif axis_name == plot_context.INDEX_AXIS:
            return limits.index_limits

        return None  # No limits set

    @staticmethod
    def finalizePlot(
        plot_context: PlotContext,
        figure: Figure,
        axes: Axes,
        default_x_label: str = "Unnamed",
        default_y_label: str = "Unnamed",
    ) -> None:
        PlotTools.showLegend(axes, plot_context)
        PlotTools.showGrid(axes, plot_context)

        PlotTools.__setupLabels(plot_context, default_x_label, default_y_label)

        plot_config = plot_context.plotConfig()
        axes.set_xlabel(plot_config.xLabel())  # type: ignore
        axes.set_ylabel(plot_config.yLabel())  # type: ignore

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
    def __setupLabels(
        plot_context: PlotContext, default_x_label: str, default_y_label: str
    ) -> None:
        config = plot_context.plotConfig()

        if config.xLabel() is None:
            config.setXLabel(default_x_label)

        if config.yLabel() is None:
            config.setYLabel(default_y_label)
