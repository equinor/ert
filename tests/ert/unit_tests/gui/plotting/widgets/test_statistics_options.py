from unittest.mock import Mock

from ert.gui.plotting.utils import PlotConfig
from ert.gui.plotting.widgets.plot_controls.statistics_options import StatisticsOptions


def test_that_statistics_options_show_mean_and_p10_p90_by_default(qtbot):
    options = StatisticsOptions(Mock())
    qtbot.addWidget(options)
    default_config = PlotConfig()

    assert options.get_statistics_style("mean").line_style == "-"
    assert options.get_statistics_style("p10-p90").line_style == "--"

    for style_name in ["p50", "std", "min-max", "p33-p67"]:
        assert (
            options.get_statistics_style(style_name).line_style
            == default_config.get_statistics_style(style_name).line_style
        )

    assert (
        options.get_standard_deviation_factor()
        == default_config.get_standard_deviation_factor()
    )
    assert (
        options.is_distribution_line_enabled()
        == default_config.is_distribution_line_enabled()
    )


def test_that_changing_statistics_style_invokes_update_callback(qtbot):
    connection_point = Mock()
    options = StatisticsOptions(connection_point)
    qtbot.addWidget(options)

    options._style_edits["mean"]._style_chooser.marker_chooser.setCurrentIndex(1)

    assert options.get_statistics_style("mean").marker == "x"
    connection_point.assert_called_once()


def test_that_statistics_and_distribution_visibility_can_be_controlled(qtbot):
    options = StatisticsOptions(Mock())
    qtbot.addWidget(options)

    options.set_statistics_available(False)
    options.set_distribution_lines_available(False)
    assert options._statistics_controls.isHidden()
    assert options._distribution_lines.isHidden()

    options.set_statistics_available(True)
    options.set_distribution_lines_available(True)
    assert not options._statistics_controls.isHidden()
    assert not options._distribution_lines.isHidden()


def test_that_statistics_presets_update_styles(qtbot):
    options = StatisticsOptions(Mock())
    qtbot.addWidget(options)

    options._presets.setCurrentIndex(3)

    assert options.get_statistics_style("mean").line_style == "-"
    assert options.get_statistics_style("p50").line_style == "--"
    assert options.get_statistics_style("p10-p90").line_style == "area"


def test_that_apply_to_plot_config_transfers_statistics_settings(qtbot):
    options = StatisticsOptions(Mock())
    qtbot.addWidget(options)

    options._presets.setCurrentIndex(1)
    options._std_dev_factor.setValue(3)
    options._distribution_lines.setChecked(True)

    config = PlotConfig()
    options.apply_to_plot_config(config)

    assert config.get_statistics_style("mean").line_style == "-"
    assert config.get_statistics_style("mean").marker == "o"
    assert config.get_statistics_style("std").line_style == "--"
    assert config.get_statistics_style("std").marker == "D"
    assert config.get_standard_deviation_factor() == 3
    assert config.is_distribution_line_enabled()


def test_that_reset_restores_selected_statistics_preset(qtbot):
    connection_point = Mock()
    options = StatisticsOptions(connection_point)
    qtbot.addWidget(options)

    options._presets.setCurrentIndex(3)
    options._update_style("p10-p90", "-", None)
    options._std_dev_factor.setValue(3)
    options._distribution_lines.setChecked(True)
    connection_point.reset_mock()

    options._reset_button.click()

    default_config = PlotConfig()
    assert options.get_statistics_style("mean").line_style == "-"
    assert options.get_statistics_style("p50").line_style == "--"
    assert options.get_statistics_style("std").line_style == ":"
    assert options.get_statistics_style("min-max").line_style == "--"
    assert options.get_statistics_style("p10-p90").line_style == "area"
    assert options.get_statistics_style("p33-p67").line_style == "area"

    assert (
        options.get_standard_deviation_factor()
        == default_config.get_standard_deviation_factor()
    )
    assert (
        options.is_distribution_line_enabled()
        == default_config.is_distribution_line_enabled()
    )
    assert options._presets.currentIndex() == 3
    connection_point.assert_called_once()
