from unittest.mock import Mock

from ert.gui.plotting.widgets.plot_controls.misfits_options import MisfitsOptions


def test_that_misfits_options_has_expected_default_checkbox_states(qtbot):
    options = MisfitsOptions(Mock())
    qtbot.addWidget(options.get_widget())

    assert options.mean_checkbox_state is True
    assert options.outliers_checkbox_state is True
    assert options.box_checkbox_state is True
    assert options.scatter_checkbox_state is False


def test_that_setting_misfits_option_state_updates_underlying_checkbox(qtbot):
    options = MisfitsOptions(Mock())
    qtbot.addWidget(options.get_widget())

    options.scatter_checkbox_state = True
    options.mean_checkbox_state = False
    options.outliers_checkbox_state = False
    options.box_checkbox_state = False

    assert options.scatter_checkbox_state is True
    assert options.mean_checkbox_state is False
    assert options.outliers_checkbox_state is False
    assert options.box_checkbox_state is False


def test_that_toggling_a_misfits_checkbox_invokes_the_connection_point(qtbot):
    connection_point = Mock()
    options = MisfitsOptions(connection_point)
    qtbot.addWidget(options.get_widget())

    options.scatter_checkbox_state = not options.scatter_checkbox_state

    connection_point.assert_called()
