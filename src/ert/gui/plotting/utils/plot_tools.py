from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import matplotlib.ticker as mticker
from matplotlib.collections import PathCollection
from matplotlib.container import BarContainer
from matplotlib.lines import Line2D

from ert.gui.plotting.utils.plot_context import PlotType
from ert.gui.plotting.utils.tooltip_manager import (
    create_tooltip_manager,
)

logger = logging.getLogger(__name__)
if TYPE_CHECKING:
    from datetime import date

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from ert.gui.plotting.utils import PlotContext


class ConditionalAxisFormatter(mticker.Formatter):
    """
    Show scientific notation only when |x| < low or |x| >= high.
    Otherwise show plain numbers.

    Parameters
    ----------
    low : float
        Lower threshold for switching to scientific notation.
    high : float
        Upper threshold for switching to scientific notation.
    precision : int
        Digits after decimal in the mantissa for scientific labels.
    """

    def __init__(
        self,
        low: float = 1e-3,
        high: float = 1e4,
        precision: int = 0,
    ) -> None:
        self.low = low
        self.high = high
        self.precision = precision

    def __call__(self, x: float, pos: int | None = None) -> str:
        if (
            x == 0
            or (self.low <= abs(x) < self.high)
            or x in {math.inf, -math.inf, math.nan}
        ):
            return f"{x:.6g}"

        s = f"{x:.{self.precision}e}"
        mant, exp = s.split("e")
        mant = mant.rstrip("0").rstrip(".")  # '-2.0' -> '-2'
        sign = "-" if exp.startswith("-") else ""
        exp_num = exp.lstrip("+-").lstrip("0") or "0"

        return f"{mant}e{sign}{exp_num}"


class PlotTools:
    @staticmethod
    def showGrid(axes: Axes, plot_context: PlotContext) -> None:
        config = plot_context.plotConfig()
        if config.is_grid_enabled():
            if plot_context.plot_type == PlotType.BAR:
                axes.grid(axis="y", color="black", alpha=0.1)
            else:
                axes.grid(visible=True, color="black", alpha=0.4)

    @staticmethod
    def showLegend(axes: Axes, plot_context: PlotContext) -> None:
        config = plot_context.plotConfig()
        if config.is_legend_enabled() and len(config.legend_items()) > 0:
            axes.legend(config.legend_items(), config.legend_labels(), numpoints=1)

    @staticmethod
    def _getXAxisLimits(
        plot_context: PlotContext,
    ) -> (
        tuple[int | None, int | None]
        | tuple[float | None, float | None]
        | tuple[date | None, date | None]
        | None
    ):
        limits = plot_context.plotConfig().limits
        axis_name = plot_context.x_axis

        if axis_name == plot_context.VALUE_AXIS:
            return limits.value_limits
        if axis_name == plot_context.COUNT_AXIS:
            return None  # Histogram takes care of itself
        if axis_name == plot_context.DATE_AXIS:
            return limits.date_limits
        if axis_name == plot_context.DENSITY_AXIS:
            return limits.density_limits
        if axis_name == plot_context.INDEX_AXIS:
            return limits.index_limits

        return None  # No limits set

    @staticmethod
    def _getYAxisLimits(
        plot_context: PlotContext,
    ) -> (
        tuple[int | None, int | None]
        | tuple[float | None, float | None]
        | tuple[date | None, date | None]
        | None
    ):
        limits = plot_context.plotConfig().limits
        axis_name = plot_context.y_axis

        if axis_name == plot_context.VALUE_AXIS:
            return limits.value_limits
        if axis_name == plot_context.COUNT_AXIS:
            return None  # Histogram takes care of itself
        if axis_name == plot_context.DATE_AXIS:
            return limits.date_limits
        if axis_name == plot_context.DENSITY_AXIS:
            return limits.density_limits
        if axis_name == plot_context.INDEX_AXIS:
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
        axes.set_xlabel(plot_config.x_label())  # type: ignore
        axes.set_ylabel(plot_config.y_label())  # type: ignore

        x_axis_limits = PlotTools._getXAxisLimits(plot_context)
        if x_axis_limits is not None:
            axes.set_xlim(*x_axis_limits)

        y_axis_limits = PlotTools._getYAxisLimits(plot_context)
        if y_axis_limits is not None:
            axes.set_ylim(*y_axis_limits)

        axes.set_title(plot_config.title())

        if plot_context.is_date_support_active():
            figure.autofmt_xdate()

        for spine in ("right", "left", "top"):
            axes.spines[spine].set_visible(False)

    @staticmethod
    def __setupLabels(
        plot_context: PlotContext, default_x_label: str, default_y_label: str
    ) -> None:
        config = plot_context.plotConfig()

        if config.x_label() is None:
            config.set_x_label(default_x_label)

        if config.y_label() is None:
            config.set_y_label(default_y_label)

    @staticmethod
    def labels_on_hover(
        plot_type: PlotType,
        axes: Axes,
        figure: Figure,
        data: list[list[Line2D]] | PathCollection | list[BarContainer],
        labels: list[str],
        **options: Any,
    ) -> None:
        hover_box = axes.annotate(
            "",
            xy=(0, 0),
            xytext=(-30, 10),
            textcoords="offset points",
            bbox={"boxstyle": "round,pad=0.5", "fc": "white", "alpha": 0.7},
        )
        hover_box.set_visible(False)

        hover_color = options.get("hover_color")
        disable_values = options.get("disable_values", False)
        tooltip_manager = create_tooltip_manager(
            plot_type,
            hover_box,
            figure,
            axes,
            hover_color=hover_color,
            disable_values=disable_values,
        )

        figure.canvas.mpl_connect(
            "motion_notify_event",
            lambda event: tooltip_manager.on_hover(event, data, labels),  # type: ignore
        )
