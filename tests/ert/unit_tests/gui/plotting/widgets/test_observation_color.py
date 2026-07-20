from unittest.mock import Mock

import pytest
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QCheckBox

from ert.gui.plotting.utils.plot_config import PlotConfig
from ert.gui.plotting.widgets.plot_controls.observation_color import (
    ObservationColorEdit,
)


def _make_edit(qtbot, *, checked: bool = True):
    checkbox = QCheckBox("Observations")
    checkbox.setChecked(checked)
    connection_point = Mock()
    edit = ObservationColorEdit(
        connection_point=connection_point,
        observation_checkbox=checkbox,
    )
    qtbot.addWidget(checkbox)
    qtbot.addWidget(edit)
    return edit, checkbox, connection_point


def test_that_default_observation_color_is_opaque_black(qtbot):
    edit, _, _ = _make_edit(qtbot)

    expected_color, expected_alpha = PlotConfig().observations_color()
    color_name, alpha = edit.get_observations_color()
    assert color_name == expected_color
    assert alpha == pytest.approx(expected_alpha, abs=1e-2)


def test_that_get_observations_color_reflects_updated_color_box(qtbot):
    edit, _, _ = _make_edit(qtbot)

    edit._observations_color_box.color = ("#ff0000", 0.5)

    color_name, alpha = edit.get_observations_color()
    assert color_name == "#ff0000"
    assert alpha == pytest.approx(0.5, abs=1e-2)


def test_that_changing_color_invokes_connection_point(qtbot):
    edit, _, connection_point = _make_edit(qtbot)

    edit._observations_color_box.colorChanged.emit(QColor("#00ff00"))

    connection_point.assert_called_once()


def test_that_widget_visibility_follows_checkbox_toggle(qtbot):
    edit, checkbox, _ = _make_edit(qtbot, checked=True)
    edit.show()

    checkbox.setChecked(False)
    assert not edit.isVisible()

    checkbox.setChecked(True)
    assert edit.isVisible()


@pytest.mark.parametrize("checked", [True, False])
def test_that_initial_visibility_mirrors_checkbox_state(qtbot, checked):
    edit, _, _ = _make_edit(qtbot, checked=checked)

    assert edit.isVisibleTo(edit.parentWidget()) is checked
