from unittest.mock import Mock

from ert.gui.plotting.utils import PlotConfig
from ert.gui.plotting.utils.statistics_style import STATISTICS
from ert.gui.plotting.widgets.plot_controls.statistics_options import StatisticsOptions


def test_that_statistics_options_have_expected_default_toggle_states(qtbot):
    options = StatisticsOptions(Mock())
    qtbot.addWidget(options.get_widget())

    checked = {
        statistic
        for statistic, checkbox in options._toggles.items()
        if checkbox.isChecked()
    }

    assert checked == {"mean", "p10-p90"}
    assert not options._area_toggle.isChecked()
    assert options._std_dev_factor.value() == 1


def test_that_unchecked_statistics_are_hidden_in_the_plot_config(qtbot):
    options = StatisticsOptions(Mock())
    qtbot.addWidget(options.get_widget())
    plot_config = PlotConfig()

    options.apply_to(plot_config)

    for statistic in STATISTICS:
        style = plot_config.get_statistics_style(statistic)
        if statistic in {"mean", "p10-p90"}:
            assert style.is_visible()
        else:
            assert not style.is_visible()


def test_that_enabling_area_switches_band_statistics_to_a_filled_style(qtbot):
    options = StatisticsOptions(Mock())
    qtbot.addWidget(options.get_widget())

    line_config = PlotConfig()
    options.apply_to(line_config)
    assert line_config.get_statistics_style("p10-p90").line_style == "--"

    options._area_toggle.setChecked(True)
    area_config = PlotConfig()
    options.apply_to(area_config)
    assert area_config.get_statistics_style("p10-p90").line_style == "#"


def test_that_the_std_dev_multiplier_is_applied_to_the_plot_config(qtbot):
    options = StatisticsOptions(Mock())
    qtbot.addWidget(options.get_widget())
    options._std_dev_factor.setValue(3)
    plot_config = PlotConfig()

    options.apply_to(plot_config)

    assert plot_config.get_standard_deviation_factor() == 3


def test_that_toggling_a_statistic_invokes_the_connection_point(qtbot):
    connection_point = Mock()
    options = StatisticsOptions(connection_point)
    qtbot.addWidget(options.get_widget())

    options._toggles["p50"].setChecked(True)

    connection_point.assert_called()


def test_that_applying_options_does_not_invoke_the_connection_point(qtbot):
    connection_point = Mock()
    options = StatisticsOptions(connection_point)
    qtbot.addWidget(options.get_widget())
    connection_point.reset_mock()

    options.apply_to(PlotConfig())

    connection_point.assert_not_called()
