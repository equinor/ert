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
    assert math.isclose(style.alpha, 1.0)
    assert style.marker == ""  # noqa: PLC1901
    assert math.isclose(style.width, 1.0)
    assert math.isclose(style.size, 7.5)
    assert style.is_enabled()

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
    assert math.isclose(style.width, 0.0)

    style.size = -1
    assert math.isclose(style.size, 0.0)

    style.alpha = 1.1
    assert math.isclose(style.alpha, 1.0)

    style.alpha = -0.1
    assert math.isclose(style.alpha, 0.0)

    style.set_enabled(False)
    assert not style.is_enabled()


def test_plot_style_copy_style():
    style = PlotStyle(
        "Test", color="red", alpha=0.5, line_style=".", marker="o", width=2.5
    )
    style.set_enabled(False)

    copy_style = PlotStyle("Copy")

    copy_style.copy_style_from(style)

    assert style.name != copy_style.name
    assert style.color == copy_style.color
    assert style.alpha == copy_style.alpha
    assert style.line_style == copy_style.line_style
    assert style.marker == copy_style.marker
    assert style.width == copy_style.width
    assert style.size == copy_style.size
    assert style.is_enabled() != copy_style.is_enabled()

    another_copy_style = PlotStyle("Another Copy")
    another_copy_style.copy_style_from(style, copy_enabled_state=True)
    assert style.is_enabled() == another_copy_style.is_enabled()


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

    plot_config.set_distribution_line_enabled(True)
    plot_config.set_legend_enabled(False)
    plot_config.set_grid_enabled(False)
    plot_config.set_observations_enabled(False)

    style = PlotStyle("test_style", line_style=".", marker="g", width=2.5, size=7.5)

    plot_config.set_default_style(style)
    plot_config.set_statistics_style("mean", style)
    plot_config.set_statistics_style("min-max", style)
    plot_config.set_statistics_style("p50", style)
    plot_config.set_statistics_style("p10-p90", style)
    plot_config.set_statistics_style("p33-p67", style)
    plot_config.set_statistics_style("std", style)

    copy_of_plot_config = PlotConfig(title="Copy of Golden Sample")
    copy_of_plot_config.copy_config_from(plot_config)

    assert plot_config.is_legend_enabled() == copy_of_plot_config.is_legend_enabled()
    assert plot_config.is_grid_enabled() == copy_of_plot_config.is_grid_enabled()
    assert (
        plot_config.is_observations_enabled()
        == copy_of_plot_config.is_observations_enabled()
    )
    assert (
        plot_config.is_distribution_line_enabled()
        == copy_of_plot_config.is_distribution_line_enabled()
    )

    assert plot_config.observations_style() == copy_of_plot_config.observations_style()

    assert plot_config.histogram_style() == copy_of_plot_config.histogram_style()
    assert plot_config.default_style() == copy_of_plot_config.default_style()
    assert plot_config.current_color() == copy_of_plot_config.current_color()

    assert plot_config.get_statistics_style(
        "mean"
    ) == copy_of_plot_config.get_statistics_style("mean")
    assert plot_config.get_statistics_style(
        "min-max"
    ) == copy_of_plot_config.get_statistics_style("min-max")
    assert plot_config.get_statistics_style(
        "p50"
    ) == copy_of_plot_config.get_statistics_style("p50")
    assert plot_config.get_statistics_style(
        "p10-p90"
    ) == copy_of_plot_config.get_statistics_style("p10-p90")
    assert plot_config.get_statistics_style(
        "p33-p67"
    ) == copy_of_plot_config.get_statistics_style("p33-p67")
    assert plot_config.get_statistics_style(
        "std"
    ) == copy_of_plot_config.get_statistics_style("std")

    assert plot_config.title() == copy_of_plot_config.title()

    assert plot_config.limits == copy_of_plot_config.limits

    plot_config.current_color()  # cycle state will not be copied
    plot_config.next_color()

    copy_of_plot_config = PlotConfig(title="Another Copy of Golden Sample")
    copy_of_plot_config.copy_config_from(plot_config)

    assert plot_config.observations_style() == copy_of_plot_config.observations_style()

    assert plot_config.histogram_style() != copy_of_plot_config.histogram_style()
    assert plot_config.default_style() != copy_of_plot_config.default_style()
    assert plot_config.current_color() != copy_of_plot_config.current_color()

    assert plot_config.get_statistics_style(
        "mean"
    ) != copy_of_plot_config.get_statistics_style("mean")
    assert plot_config.get_statistics_style(
        "min-max"
    ) != copy_of_plot_config.get_statistics_style("min-max")
    assert plot_config.get_statistics_style(
        "p50"
    ) != copy_of_plot_config.get_statistics_style("p50")
    assert plot_config.get_statistics_style(
        "p10-p90"
    ) != copy_of_plot_config.get_statistics_style("p10-p90")
    assert plot_config.get_statistics_style(
        "p33-p67"
    ) != copy_of_plot_config.get_statistics_style("p33-p67")
    assert plot_config.get_statistics_style(
        "std"
    ) != copy_of_plot_config.get_statistics_style("std")
