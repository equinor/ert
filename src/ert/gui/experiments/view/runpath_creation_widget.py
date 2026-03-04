from __future__ import annotations

from PyQt6.QtCore import QSize
from PyQt6.QtGui import QMovie
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from ert.run_models._create_run_path import RunPathCreationEvent


class RunpathCreationProgressWidget(QWidget):
    """Progress bar shown as a tab while runpaths are being created."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._total: int = 1

        spin_movie = QMovie("img:loading.gif")
        spin_movie.setSpeed(60)
        spin_movie.setScaledSize(QSize(16, 16))
        spin_movie.start()
        self._spinner = QLabel()
        self._spinner.setFixedSize(QSize(16, 16))
        self._spinner.setMovie(spin_movie)

        self._label = QLabel("Preparing runpaths...")

        self._bar = QProgressBar()
        self._bar.setRange(0, 1)
        self._bar.setValue(0)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(8)
        self._bar.setStyleSheet("""
            QProgressBar {
                border: none;
                border-radius: 4px;
                background-color: rgba(128, 128, 128, 0.2);
            }
            QProgressBar::chunk {
                border-radius: 4px;
                background-color: #43A047;
            }
        """)

        spinner_row = QHBoxLayout()
        spinner_row.setSpacing(6)
        spinner_row.addStretch()
        spinner_row.addWidget(self._spinner)
        spinner_row.addWidget(self._label)
        spinner_row.addStretch()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        layout.addWidget(self._bar)
        layout.addLayout(spinner_row)
        layout.addStretch()

    def handle_event(self, event: RunPathCreationEvent) -> None:
        if event.sub_type == "StartingTotalRunPathCreation":
            self._total = (
                1
                if event.total_runpaths_to_create is None
                else event.total_runpaths_to_create
            )
            self._bar.setRange(0, self._total)
            self._bar.setValue(0)
            self._label.setText(f"0 / {self._total} runpaths created")

        elif event.sub_type == "TotalRunPathCreationUpdate":
            if event.created_runpaths_count is not None:
                self._bar.setValue(event.created_runpaths_count)
                self._label.setText(
                    f"{event.created_runpaths_count} / {self._total} runpaths created"
                )
