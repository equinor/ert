from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ert.gui.experiments.view import DiskSpaceWidget
from ert.gui.experiments.view.disk_space_widget import (
    CRITICAL_RED,
    NORMAL_GREEN,
    WARNING_YELLOW,
    MountType,
)


def test_disk_space_widget(qtbot):
    disk_space_widget = DiskSpaceWidget(Path("/tmp"))
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


@pytest.mark.parametrize(
    ("mount_path", "mount_type"),
    [
        ("/tmp", MountType.RUNPATH),
        ("/usr", MountType.STORAGE),
    ],
)
def test_disk_space_widget_label_content(qtbot, mount_path: str, mount_type: MountType):
    disk_space_widget = DiskSpaceWidget(Path(mount_path), mount_type)
    qtbot.addWidget(disk_space_widget)
    disk_space_widget.update_status()

    assert disk_space_widget.isVisible()
    assert mount_path in disk_space_widget.mount_point_label.text()
    assert mount_type.name.capitalize() in disk_space_widget.mount_point_label.text()
