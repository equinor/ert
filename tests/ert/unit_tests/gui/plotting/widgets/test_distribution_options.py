import logging
from unittest.mock import Mock

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QCheckBox, QRadioButton

from ert.gui.plotting.widgets.plot_controls import DistributionOptions


def find_and_click_checkbox(
    widget, obj_name, qtbot, qt_type: type[QCheckBox] | type[QRadioButton]
):
    child = widget.findChild(qt_type, obj_name)
    assert child is not None
    qtbot.mouseClick(child, Qt.MouseButton.LeftButton)


def test_that_distribution_options_has_expected_default_checkbox_states(qtbot):
    options = DistributionOptions(Mock())
    qtbot.addWidget(options.get_widget())

    assert options.histogram_checkbox_state is True
    assert options.gkde_checkbox_state is True
    assert options.rug_checkbox_state is True
    assert options.histogram_by_density is True


def test_that_setting_distribution_option_state_updates_underlying_checkbox(qtbot):
    options = DistributionOptions(Mock())
    qtbot.addWidget(options.get_widget())

    options.histogram_checkbox_state = False
    options.gkde_checkbox_state = False
    options.rug_checkbox_state = False
    options.histogram_by_density = False

    assert options.histogram_checkbox_state is False
    assert options.gkde_checkbox_state is False
    assert options.rug_checkbox_state is False
    assert options.histogram_by_density is False


def test_that_unchecking_each_checkbox_unchecks_it_and_invokes_connection_point(
    qtbot,
):
    connection_point = Mock()
    options = DistributionOptions(connection_point)
    widget = options.get_widget()
    qtbot.addWidget(widget)
    widget.show()

    for checkbox_name in ["histogram_checkbox", "gkde_checkbox", "rug_checkbox"]:
        find_and_click_checkbox(widget, checkbox_name, qtbot, QCheckBox)
        assert widget.findChild(QCheckBox, checkbox_name).isChecked() is False
        connection_point.assert_called()


def test_that_toggling_a_distribution_checkbox_invokes_the_connection_point(qtbot):
    connection_point = Mock()
    options = DistributionOptions(connection_point)
    widget = options.get_widget()
    qtbot.addWidget(widget)
    widget.show()

    find_and_click_checkbox(widget, "gkde_checkbox", qtbot, QCheckBox)

    connection_point.assert_called()


def test_that_histogram_options_are_hidden_when_histogram_is_unchecked(qtbot):
    options = DistributionOptions(Mock())
    widget = options.get_widget()
    qtbot.addWidget(widget)
    widget.show()

    assert options._histogram_options_widget.isVisible() is True

    find_and_click_checkbox(widget, "histogram_checkbox", qtbot, QCheckBox)

    assert options._histogram_options_widget.isVisible() is False


def test_that_turning_histogram_off_resets_mode_to_by_density(qtbot):
    options = DistributionOptions(Mock())
    widget = options.get_widget()
    qtbot.addWidget(widget)
    widget.show()

    assert options.histogram_by_density is True

    find_and_click_checkbox(widget, "by_count_radiobutton", qtbot, QRadioButton)
    assert options.histogram_by_density is False

    find_and_click_checkbox(widget, "histogram_checkbox", qtbot, QCheckBox)

    assert options.histogram_by_density is True


def test_that_a_distribution_toggle_is_logged_only_once_per_session(qtbot, caplog):
    caplog.set_level(
        logging.INFO,
        logger="ert.gui.plotting.widgets.plot_controls.distribution_options",
    )

    options = DistributionOptions(Mock())
    widget = options.get_widget()
    qtbot.addWidget(widget)
    widget.show()

    find_and_click_checkbox(widget, "gkde_checkbox", qtbot, QCheckBox)
    find_and_click_checkbox(widget, "gkde_checkbox", qtbot, QCheckBox)
    find_and_click_checkbox(widget, "gkde_checkbox", qtbot, QCheckBox)

    gkde_logs = [
        r.message
        for r in caplog.records
        if r.message == "Plot sidebar option used: 'Show Gaussian KDE'"
    ]
    assert len(gkde_logs) == 1


def test_that_each_distribution_toggle_is_logged_independently(qtbot, caplog):
    caplog.set_level(
        logging.INFO,
        logger="ert.gui.plotting.widgets.plot_controls.distribution_options",
    )

    options = DistributionOptions(Mock())
    widget = options.get_widget()
    qtbot.addWidget(widget)
    widget.show()

    for name, checkbox_class in (
        ("gkde_checkbox", QCheckBox),
        ("rug_checkbox", QCheckBox),
        ("by_count_radiobutton", QRadioButton),
        ("by_density_radiobutton", QRadioButton),
        ("histogram_checkbox", QCheckBox),
    ):
        find_and_click_checkbox(widget, name, qtbot, checkbox_class)

    for option_name in [
        "Show histogram",
        "Y axis by density",
        "Y axis by count",
        "Show Gaussian KDE",
        "Show rug plot",
    ]:
        assert f"Plot sidebar option used: '{option_name}'" in caplog.text
