from qtpy.QtCore import Qt
from qtpy.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QWidget

from ert.shared.status.utils import disk_space_status


class DiskSpaceWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # Text label
        self.label = QLabel(self)

        # Progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedWidth(100)
        self.progress_bar.setAlignment(Qt.AlignCenter)

        # Add color styling based on usage
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #ccc;
                border-radius: 2px;
                text-align: center;
            }

            QProgressBar::chunk {
                background-color: qlineargradient(
                    x1: 0, y1: 0.5, x2: 1, y2: 0.5,
                    stop: 0 #2ecc71,
                    stop: 0.7 #f1c40f,
                    stop: 0.9 #e74c3c
                );
            }
        """)

        layout.addWidget(self.label)
        layout.addWidget(self.progress_bar)

    def update_status(self, run_path_mp):
        """Update both the label and progress bar with current disk usage"""
        disk_usage = disk_space_status(run_path_mp)
        if disk_usage is not None:
            self.label.setText("Disk space used runpath:")
            self.progress_bar.setValue(int(disk_usage))
            self.progress_bar.setFormat(f"{disk_usage:.1f}%")
            self.setVisible(True)
        else:
            self.setVisible(False)
