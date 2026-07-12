# ruff: noqa: E402

# At least for some combinations of pandas and matplotlib the numpy.datetime64
# dates coming from pandas are not correctly recognized/converted by matplotlib.
# Calling this converter.register() method seems to fix the problem.
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

from .plot_config import PlotConfig
from .plot_config_factory import PlotConfigFactory
from .plot_config_history import PlotConfigHistory
from .plot_context import PlotContext
from .plot_limits import PlotLimits
from .plot_style import PlotStyle
from .plot_tools import ConditionalAxisFormatter, PlotTools
from .plot_types import ObservationPlotLocations
from .tooltip_manager import (
    BarTooltipManager,
    LineTooltipManager,
    ScatterTooltipManager,
    ToolTipManager,
)

__all__ = [
    "BarTooltipManager",
    "ConditionalAxisFormatter",
    "LineTooltipManager",
    "ObservationPlotLocations",
    "PlotConfig",
    "PlotConfigFactory",
    "PlotConfigHistory",
    "PlotContext",
    "PlotLimits",
    "PlotStyle",
    "PlotTools",
    "ScatterTooltipManager",
    "ToolTipManager",
]
