from unittest.mock import Mock

import pytest
from PyQt6.QtWidgets import QCheckBox

from ert.gui.plotting.widgets.plot_controls.map_options import MapOptions


def test_that_map_options_has_expected_default_states(qtbot):
    options = MapOptions(Mock())
    qtbot.addWidget(options.get_widget())

    assert options.misfit_map_checkbox_state is True
    assert options.observations_map_checkbox_state is True


@pytest.mark.parametrize(
    "checkbox_name", ["misfit_map_checkbox", "observations_map_checkbox"]
)
def test_that_toggling_a_map_checkbox_invokes_the_connection_point(
    qtbot, checkbox_name
):
    connection_point = Mock()
    options = MapOptions(connection_point)
    widget = options.get_widget()
    qtbot.addWidget(widget)
    widget.show()

    checkbox = widget.findChild(QCheckBox, checkbox_name)
    assert checkbox is not None
    assert checkbox.isVisible()

    checkbox.click()

    connection_point.assert_called_once()
