import contextlib
import shutil
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QWidget

from ert.shared.status.utils import byte_with_unit

CRITICAL_RED = "#e74c3c"
WARNING_YELLOW = "#f1c40f"
NORMAL_GREEN = "#2ecc71"


class DiskSpaceWidget(QWidget):
    def __init__(self, mount_path: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.mount_path = mount_path

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # Text label
        self.usage_label = QLabel(self)
        self.space_left_label = QLabel(self)

        # Progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedWidth(100)
        self.progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self.usage_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.space_left_label)

    def _get_status(self) -> tuple[float, str] | None:
        with contextlib.suppress(Exception):
            disk_info = shutil.disk_usage(self.mount_path)
            percentage_used = (disk_info.used / disk_info.total) * 100
            return percentage_used, byte_with_unit(disk_info.free)
        return None

    def update_status(self) -> None:
        """Update both the label and progress bar with current disk usage"""
        if (disk_info := self._get_status()) is not None:
            usage, space_left = disk_info
            self.usage_label.setText("Disk space runpath:")
            self.progress_bar.setValue(int(usage))
            self.progress_bar.setFormat(f"{usage:.1f}%")

            # Set color based on usage threshold
            if usage >= 90:
                color = CRITICAL_RED
            elif usage >= 70:
                color = WARNING_YELLOW
            else:
                color = NORMAL_GREEN

            self.progress_bar.setStyleSheet(f"""
                QProgressBar {{
                    border: 1px solid #ccc;
                    border-radius: 2px;
                    text-align: center;
                }}

                QProgressBar::chunk {{
                    background-color: {color};
                }}
            """)

            self.space_left_label.setText(f"{space_left} free")

            self.setVisible(True)
        else:
            self.setVisible(False)
