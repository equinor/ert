from unittest.mock import Mock

import pytest

from ert.gui.plotting.widgets.plot_controls.general_options import GeneralOptions


def test_that_general_options_has_expected_default_checkbox_states(qtbot):
    options = GeneralOptions(Mock(), is_everest=False)
    qtbot.addWidget(options.get_widget())

    assert options.legend_checkbox_state is True
    assert options.grid_checkbox_state is True
    assert options.history_checkbox_state is True
    assert options.observations_checkbox_state is True
    assert options.log_checkbox_state is False


@pytest.mark.parametrize(
    ("option_name", "expected_state"),
    [
        pytest.param("legend", 0),
        pytest.param("grid", 0),
        pytest.param("history", 0),
        pytest.param("observations", 0),
        pytest.param("log", 2),
    ],
)
def test_that_toggling_a_general_option_invokes_the_connection_point(
    qtbot, option_name, expected_state
):
    connection_point = Mock()
    options = GeneralOptions(connection_point, is_everest=False)
    qtbot.addWidget(options.get_widget())

    checkbox_name = (
        "_toggle_log_scale" if option_name == "log" else f"_toggle_{option_name}"
    )
    checkbox = getattr(options, checkbox_name)
    checkbox.setVisible(True)
    checkbox.setChecked(not checkbox.isChecked())

    connection_point.assert_called_once_with(expected_state)


@pytest.mark.parametrize(
    ("history_visible", "observations_visible"),
    [
        pytest.param(False, False, id="neither-visible"),
        pytest.param(True, False, id="history-visible"),
        pytest.param(False, True, id="observations-visible"),
        pytest.param(True, True, id="both-visible"),
    ],
)
def test_that_history_and_observations_visibility_can_be_set(
    qtbot,
    history_visible,
    observations_visible,
):
    options = GeneralOptions(Mock(), is_everest=False)
    widget = options.get_widget()
    qtbot.addWidget(widget)
    widget.show()

    options.set_history_visible(history_visible)
    options.set_observations_visible(observations_visible)

    assert options._toggle_history.isVisible() is history_visible
    assert options._toggle_observations.isVisible() is observations_visible


def test_that_everest_general_options_omit_history_and_observations(qtbot):
    options = GeneralOptions(Mock(), is_everest=True)
    qtbot.addWidget(options.get_widget())

    assert not options._toggle_history.isVisible()
    assert not options._toggle_observations.isVisible()
