import logging
from unittest.mock import Mock

import pytest
from PyQt6.QtWidgets import QCheckBox

from ert.gui.plotting.widgets.plot_controls.distribution_options import (
    NAME_AND_TOOLTIP,
    DistributionOptions,
)


def find_and_click_checkbox(widget, obj_name, qtbot, qt_type: type[QCheckBox]):
    child = widget.findChild(qt_type, obj_name)
    assert child is not None
    child.click()


def test_that_all_distribution_options_are_enabled_by_default(qtbot):
    options = DistributionOptions(Mock())
    widget = options.get_widget()
    qtbot.addWidget(widget)
    widget.show()

    assert options.histogram_checkbox_state is True
    assert options.gkde_checkbox_state is True
    assert options.rug_checkbox_state is True


@pytest.mark.parametrize(
    ("checkbox_name", "state_attr", "index"),
    [
        (NAME_AND_TOOLTIP[0][0], "histogram_checkbox_state", 0),
        (NAME_AND_TOOLTIP[1][0], "gkde_checkbox_state", 1),
        (NAME_AND_TOOLTIP[2][0], "rug_checkbox_state", 2),
    ],
)
def test_that_unchecking_a_distribution_option_updates_state_and_notifies_and_logs(
    qtbot, caplog, checkbox_name, state_attr, index
):
    caplog.set_level(
        logging.INFO,
        logger="ert.gui.plotting.widgets.plot_controls.distribution_options",
    )
    connection_point = Mock()
    options = DistributionOptions(connection_point)
    widget = options.get_widget()
    qtbot.addWidget(widget)
    widget.show()

    find_and_click_checkbox(widget, checkbox_name, qtbot, QCheckBox)

    assert widget.findChild(QCheckBox, checkbox_name).isChecked() is False
    assert getattr(options, state_attr) is False
    connection_point.assert_called_once()
    assert (
        f"Plot sidebar option used: 'Distribution option: {NAME_AND_TOOLTIP[index][1]}'"
        in caplog.text
    )


def test_that_a_distribution_toggle_is_logged_only_once_per_session(qtbot, caplog):
    caplog.set_level(
        logging.INFO,
        logger="ert.gui.plotting.widgets.plot_controls.distribution_options",
    )

    options = DistributionOptions(Mock())
    widget = options.get_widget()
    qtbot.addWidget(widget)
    widget.show()

    find_and_click_checkbox(widget, NAME_AND_TOOLTIP[1][0], qtbot, QCheckBox)
    find_and_click_checkbox(widget, NAME_AND_TOOLTIP[1][0], qtbot, QCheckBox)
    find_and_click_checkbox(widget, NAME_AND_TOOLTIP[1][0], qtbot, QCheckBox)

    gkde_logs = [
        r.message
        for r in caplog.records
        if r.message
        == f"Plot sidebar option used: 'Distribution option: {NAME_AND_TOOLTIP[1][1]}'"
    ]
    assert len(gkde_logs) == 1
