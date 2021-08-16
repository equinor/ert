# At least for some combinations of pandas and matplotlib the numpy.datetime64
# dates coming from pandas are not correctly recognized/converted by matplotlib.
# Calling this converter.register() method seems to fix the problem.
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

from .plot_style import PlotStyle
from .plot_limits import PlotLimits
from .plot_config import PlotConfig
from .plot_context import PlotContext
from .plot_config_history import PlotConfigHistory
from .plot_config_factory import PlotConfigFactory
