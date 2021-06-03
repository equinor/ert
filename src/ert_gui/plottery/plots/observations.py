import math


def plotObservations(observation_data, plot_context, axes):
    key = plot_context.key()
    config = plot_context.plotConfig()
    case_list = plot_context.cases()

    if (
        config.isObservationsEnabled()
        and len(case_list) > 0
        and observation_data is not None
        and not observation_data.empty
    ):
        _plotObservations(axes, config, observation_data, value_column=key)


def _plotObservations(axes, plot_config, data, value_column):
    """
    Observations are always plotted on top. z-order set to 1000

    Since it is not possible to apply different linestyles to the errorbar, the line_style / fmt is used to toggle
    visibility of the solid errorbar, by using the elinewidth  parameter.

    @type axes: matplotlib.axes.Axes
    @type plot_config: PlotConfig
    @type data: DataFrame
    @type value_column: Str
    """

    style = plot_config.observationsStyle()

    # adjusting the top and bottom bar, according to the line width/thickness
    def cap_size(line_with):
        return 0 if line_with == 0 else math.log(line_with, 1.2) + 3

    # line style set to 'off' toggles errorbar visibility
    if style.line_style == "":
        style.width = 0

    errorbars = axes.errorbar(
        x=data.columns.get_level_values("key_index").values,
        y=data.loc["OBS"].values,
        yerr=data.loc["STD"].values,
        fmt=style.line_style,
        ecolor=style.color,
        color=style.color,
        capsize=cap_size(style.width),
        capthick=style.width,  # same as width/thickness on error line
        alpha=style.alpha,
        linewidth=0,
        marker=style.marker,
        ms=style.size,
        elinewidth=style.width,
        zorder=1000,
    )
