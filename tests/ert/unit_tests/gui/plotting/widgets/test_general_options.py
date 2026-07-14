from unittest.mock import Mock

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QCheckBox

from ert.gui.plotting.widgets.plot_controls.general_options import GeneralPlotOptions


def test_that_general_options_has_expected_default_checkbox_states(qtbot):
    options = GeneralPlotOptions(Mock(), is_everest=False)
    qtbot.addWidget(options.get_widget())

    assert options.legend_checkbox_state is True
    assert options.grid_checkbox_state is True
    assert options.history_checkbox_state is True
    assert options.observations_checkbox_state is True
    assert options.log_checkbox_state is False


@pytest.mark.parametrize(
    "checkbox_name",
    [
        "legend_checkbox",
        "grid_checkbox",
        "history_checkbox",
        "observations_checkbox",
        "log_scale_checkbox",
    ],
)
def test_that_toggling_a_general_option_invokes_the_connection_point2(
    qtbot,
    checkbox_name,
) -> None:
    connection_point = Mock()
    options = GeneralPlotOptions(connection_point, is_everest=False)
    widget = options.get_widget()
    qtbot.addWidget(widget)
    widget.show()

    checkbox = widget.findChild(QCheckBox, checkbox_name)
    if checkbox_name == "log_scale_checkbox":
        checkbox.setVisible(True)
    assert checkbox is not None
    assert checkbox.isVisible()

    qtbot.mouseClick(checkbox, Qt.MouseButton.LeftButton)

    connection_point.assert_called_once()


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
    options = GeneralPlotOptions(Mock(), is_everest=False)
    qtbot.addWidget(options.get_widget())
    options.get_widget().show()

    options.set_history_visible(history_visible)
    options.set_observations_visible(observations_visible)

    assert options._toggle_history.isVisible() is history_visible
    assert options._toggle_observations.isVisible() is observations_visible


def test_that_everest_general_options_omit_history_and_observations(qtbot):
    options = GeneralPlotOptions(Mock(), is_everest=True)
    qtbot.addWidget(options.get_widget())

    assert not options._toggle_history.isVisible()
    assert not options._toggle_observations.isVisible()
