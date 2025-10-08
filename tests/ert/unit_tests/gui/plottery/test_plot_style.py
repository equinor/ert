import datetime
import math

from ert.gui.tools.plot.plottery import PlotConfig, PlotLimits, PlotStyle
from ert.gui.tools.plot.plottery.plots.plot_tools import ConditionalAxisFormatter


def test_conditional_axis_formatter():
    fmt = ConditionalAxisFormatter(low=1e-3, high=1e4, precision=0)

    assert fmt(0) == "0"
    assert fmt(-0) == "0"
    assert fmt(0.001) == "0.001"
    assert fmt(-0.001) == "-0.001"
    assert fmt(0.0001) == "1e-4"
    assert fmt(-0.0001) == "-1e-4"
    assert fmt(123) == "123"
    assert fmt(1e3) == "1000"

    assert fmt(2e-2) == "0.02"
    assert fmt(2e-6) == "2e-6"
    assert fmt(-2e-6) == "-2e-6"
    assert fmt(2e5) == "2e5"
    assert fmt(-3e6) == "-3e6"
    assert fmt(10000) == "1e4"
    assert fmt(-10000) == "-1e4"
    assert fmt(1e4) == "1e4"
    assert fmt(9999) == "9999"
    assert fmt(-9999) == "-9999"

    assert fmt(0.001) == "0.001"

    assert fmt(0.0001) == "1e-4"

    assert fmt(math.nan) == "nan"
    assert fmt(math.inf) == "inf"
    assert fmt(-math.inf) == "-inf"

    # precision
    fmt_p1 = ConditionalAxisFormatter(low=1e-3, high=1e4, precision=1)
    assert fmt_p1(2.5e-6) == "2.5e-6"


def test_plot_style_test_defaults():
    style = PlotStyle("Test")

    assert style.name == "Test"
    assert style.color == "#000000"
    assert style.line_style == "-"
    assert style.alpha == 1.0
    assert style.marker == ""  # noqa: PLC1901
    assert style.width == 1.0
    assert style.size == 7.5
    assert style.isEnabled()

    style.line_style = None
    style.marker = None

    assert style.line_style == ""  # noqa: PLC1901
    assert style.marker == ""  # noqa: PLC1901


def test_plot_style_builtin_checks():
    style = PlotStyle("Test")

    style.name = None
    assert style.name is None

    style.color = "notacolor"
    assert style.color == "notacolor"  # maybe make this a proper check in future ?

    style.line_style = None
    assert style.line_style == ""  # noqa: PLC1901

    style.marker = None
    assert style.marker == ""  # noqa: PLC1901

    style.width = -1
    assert style.width == 0.0

    style.size = -1
    assert style.size == 0.0

    style.alpha = 1.1
    assert style.alpha == 1.0

    style.alpha = -0.1
    assert style.alpha == 0.0

    style.setEnabled(False)
    assert not style.isEnabled()


def test_plot_style_copy_style():
    style = PlotStyle("Test", "red", 0.5, ".", "o", 2.5)
    style.setEnabled(False)

    copy_style = PlotStyle("Copy")

    copy_style.copyStyleFrom(style)

    assert style.name != copy_style.name
    assert style.color == copy_style.color
    assert style.alpha == copy_style.alpha
    assert style.line_style == copy_style.line_style
    assert style.marker == copy_style.marker
    assert style.width == copy_style.width
    assert style.size == copy_style.size
    assert style.isEnabled() != copy_style.isEnabled()

    another_copy_style = PlotStyle("Another Copy")
    another_copy_style.copyStyleFrom(style, copy_enabled_state=True)
    assert style.isEnabled() == another_copy_style.isEnabled()


def test_plot_config():
    plot_config = PlotConfig(title="Golden Sample", x_label="x", y_label="y")

    limits = PlotLimits()
    limits.count_limits = 1, 2
    limits.density_limits = 5, 6
    limits.date_limits = datetime.date(2005, 2, 5), datetime.date(2006, 2, 6)
    limits.index_limits = 7, 8
    limits.value_limits = 9.0, 10.0

    plot_config.limits = limits
    assert plot_config.limits == limits

    plot_config.setDistributionLineEnabled(True)
    plot_config.setLegendEnabled(False)
    plot_config.setGridEnabled(False)
    plot_config.setObservationsEnabled(False)

    style = PlotStyle("test_style", line_style=".", marker="g", width=2.5, size=7.5)

    plot_config.setDefaultStyle(style)
    plot_config.setStatisticsStyle("mean", style)
    plot_config.setStatisticsStyle("min-max", style)
    plot_config.setStatisticsStyle("p50", style)
    plot_config.setStatisticsStyle("p10-p90", style)
    plot_config.setStatisticsStyle("p33-p67", style)
    plot_config.setStatisticsStyle("std", style)

    copy_of_plot_config = PlotConfig(title="Copy of Golden Sample")
    copy_of_plot_config.copyConfigFrom(plot_config)

    assert plot_config.isLegendEnabled() == copy_of_plot_config.isLegendEnabled()
    assert plot_config.isGridEnabled() == copy_of_plot_config.isGridEnabled()
    assert (
        plot_config.isObservationsEnabled()
        == copy_of_plot_config.isObservationsEnabled()
    )
    assert (
        plot_config.isDistributionLineEnabled()
        == copy_of_plot_config.isDistributionLineEnabled()
    )

    assert plot_config.observationsStyle() == copy_of_plot_config.observationsStyle()

    assert plot_config.histogramStyle() == copy_of_plot_config.histogramStyle()
    assert plot_config.defaultStyle() == copy_of_plot_config.defaultStyle()
    assert plot_config.currentColor() == copy_of_plot_config.currentColor()

    assert plot_config.getStatisticsStyle(
        "mean"
    ) == copy_of_plot_config.getStatisticsStyle("mean")
    assert plot_config.getStatisticsStyle(
        "min-max"
    ) == copy_of_plot_config.getStatisticsStyle("min-max")
    assert plot_config.getStatisticsStyle(
        "p50"
    ) == copy_of_plot_config.getStatisticsStyle("p50")
    assert plot_config.getStatisticsStyle(
        "p10-p90"
    ) == copy_of_plot_config.getStatisticsStyle("p10-p90")
    assert plot_config.getStatisticsStyle(
        "p33-p67"
    ) == copy_of_plot_config.getStatisticsStyle("p33-p67")
    assert plot_config.getStatisticsStyle(
        "std"
    ) == copy_of_plot_config.getStatisticsStyle("std")

    assert plot_config.title() == copy_of_plot_config.title()

    assert plot_config.limits == copy_of_plot_config.limits

    plot_config.currentColor()  # cycle state will not be copied
    plot_config.nextColor()

    copy_of_plot_config = PlotConfig(title="Another Copy of Golden Sample")
    copy_of_plot_config.copyConfigFrom(plot_config)

    assert plot_config.observationsStyle() == copy_of_plot_config.observationsStyle()

    assert plot_config.histogramStyle() != copy_of_plot_config.histogramStyle()
    assert plot_config.defaultStyle() != copy_of_plot_config.defaultStyle()
    assert plot_config.currentColor() != copy_of_plot_config.currentColor()

    assert plot_config.getStatisticsStyle(
        "mean"
    ) != copy_of_plot_config.getStatisticsStyle("mean")
    assert plot_config.getStatisticsStyle(
        "min-max"
    ) != copy_of_plot_config.getStatisticsStyle("min-max")
    assert plot_config.getStatisticsStyle(
        "p50"
    ) != copy_of_plot_config.getStatisticsStyle("p50")
    assert plot_config.getStatisticsStyle(
        "p10-p90"
    ) != copy_of_plot_config.getStatisticsStyle("p10-p90")
    assert plot_config.getStatisticsStyle(
        "p33-p67"
    ) != copy_of_plot_config.getStatisticsStyle("p33-p67")
    assert plot_config.getStatisticsStyle(
        "std"
    ) != copy_of_plot_config.getStatisticsStyle("std")
