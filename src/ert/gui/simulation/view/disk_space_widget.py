from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QWidget

from ert.shared.status.utils import disk_space_status


class DiskSpaceWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

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
        self.progress_bar.setAlignment(Qt.AlignCenter)  # type: ignore

        layout.addWidget(self.usage_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.space_left_label)

    def update_status(self, mount_dir: Path) -> None:
        """Update both the label and progress bar with current disk usage"""
        disk_info = disk_space_status(mount_dir)
        if disk_info is not None:
            usage = int(disk_info[0])
            self.usage_label.setText("Disk space runpath:")
            self.progress_bar.setValue(usage)
            self.progress_bar.setFormat(f"{disk_info[0]:.1f}%")

            # Set color based on usage threshold
            if usage >= 90:
                color = "#e74c3c"  # Red for critical usage
            elif usage >= 70:
                color = "#f1c40f"  # Yellow for warning
            else:
                color = "#2ecc71"  # Green for normal usage

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

            self.space_left_label.setText(f"{disk_info[1]} free")

            self.setVisible(True)
        else:
            self.setVisible(False)
