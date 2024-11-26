from unittest.mock import MagicMock

from ert.gui.simulation.view import DiskSpaceWidget
from ert.gui.simulation.view.disk_space_widget import (
    CRITICAL_RED,
    NORMAL_GREEN,
    WARNING_YELLOW,
)


def test_disk_space_widget(qtbot):
    disk_space_widget = DiskSpaceWidget("/tmp")
    qtbot.addWidget(disk_space_widget)
    disk_space_widget._get_status = MagicMock()

    disk_space_widget._get_status.return_value = (20.0, "2 jiggabytes")
    disk_space_widget.update_status()
    assert disk_space_widget.space_left_label.text() == "2 jiggabytes free"
    assert disk_space_widget.progress_bar.value() == 20
    assert NORMAL_GREEN in disk_space_widget.progress_bar.styleSheet()

    disk_space_widget._get_status.return_value = (88.0, "2mb")
    disk_space_widget.update_status()
    assert disk_space_widget.space_left_label.text() == "2mb free"
    assert disk_space_widget.progress_bar.value() == 88
    assert WARNING_YELLOW in disk_space_widget.progress_bar.styleSheet()

    disk_space_widget._get_status.return_value = (99.9, "2 bytes")
    disk_space_widget.update_status()
    assert disk_space_widget.space_left_label.text() == "2 bytes free"
    assert disk_space_widget.progress_bar.value() == 99
    assert CRITICAL_RED in disk_space_widget.progress_bar.styleSheet()

    disk_space_widget._get_status.return_value = None
    disk_space_widget.update_status()
    assert not disk_space_widget.isVisible()
