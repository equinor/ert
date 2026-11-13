from __future__ import annotations

from typing import override

from PyQt6.QtGui import QColor, QResizeEvent
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QVBoxLayout,
)

from ert.ensemble_evaluator.state import (
    ENSEMBLE_STATE_FAILED,
    REAL_STATE_TO_COLOR,
    REALIZATION_STATE_WAITING,
)


class ProgressWidget(QFrame):
    def __init__(self) -> None:
        super().__init__()
        self.setFixedHeight(70)

        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(0, 0, 0, 0)
        self._main_layout.setSpacing(2)
        self.setLayout(self._main_layout)

        self._waiting_progress_bar = QProgressBar(self)
        self._waiting_progress_bar.setRange(0, 0)
        self._waiting_progress_bar.setFixedHeight(30)
        self._main_layout.addWidget(self._waiting_progress_bar)

        self._progress_frame = QFrame(self)
        self._main_layout.addWidget(self._progress_frame)

        self._horizontal_layout = QHBoxLayout(self._progress_frame)
        self._horizontal_layout.setContentsMargins(0, 0, 0, 0)
        self._horizontal_layout.setSpacing(0)
        self._progress_frame.setLayout(self._horizontal_layout)

        self._legend_frame = QFrame(self)
        self._main_layout.addWidget(self._legend_frame)
        self._legend_frame.setFixedHeight(24)
        self._legend_layout = QHBoxLayout(self._legend_frame)
        self._legend_layout.setContentsMargins(0, 0, 0, 0)
        self._legend_layout.setSpacing(6)

        self._status_counts: dict[str, int] = {}
        self._realization_count = 0
        self._progress_segments: dict[str, QLabel] = {}
        self._legend_labels = {}
        for realization_status, status_color_rgb in REAL_STATE_TO_COLOR.items():
            status_color_hex = QColor(*status_color_rgb).name()

            progress_segment = QLabel(self)
            progress_segment.setVisible(False)
            progress_segment.setObjectName(f"progress_{realization_status}")
            progress_segment.setStyleSheet(f"background-color : {status_color_hex}")
            self._progress_segments[realization_status] = progress_segment
            self._horizontal_layout.addWidget(progress_segment)

            if realization_status == REALIZATION_STATE_WAITING:
                marker_style = (
                    "background-color: transparent;"
                    "border-radius: 7px;"
                    f"border: 2px solid {status_color_hex};"
                )
            else:
                marker_style = (
                    f"background-color: {status_color_hex};border-radius: 7px;"
                )

            legend_marker = QLabel(self)
            legend_marker.setFixedSize(14, 14)
            legend_marker.setStyleSheet(marker_style)
            self._legend_layout.addWidget(legend_marker)

            legend_label = QLabel(self)
            legend_label.setObjectName(f"progress_label_text_{realization_status}")
            legend_label.setText(f"{realization_status} ({0}/{0})")
            self._legend_labels[realization_status] = legend_label
            self._legend_layout.addWidget(legend_label)
            self._legend_layout.addSpacing(16)

        self._legend_layout.addStretch()

    def repaint_components(self) -> None:
        if self._realization_count > 0:
            full_width = self.width()
            self.stop_waiting_progress_bar()

            for realization_status, progress_segment in self._progress_segments.items():
                progress_segment.setVisible(True)
                count = self._status_counts.get(realization_status, 0)
                width = int((count / self._realization_count) * full_width)
                progress_segment.setFixedWidth(width)

            for realization_status, legend_label in self._legend_labels.items():
                legend_label.setText(
                    f"{realization_status} "
                    f"({self._status_counts.get(realization_status, 0)}/"
                    f"{self._realization_count})"
                )

    def stop_waiting_progress_bar(self) -> None:
        self._waiting_progress_bar.setVisible(False)

    def start_waiting_progress_bar(self) -> None:
        self._waiting_progress_bar.setVisible(True)

    def set_all_failed(self) -> None:
        self.stop_waiting_progress_bar()
        full_width = self.width()
        for realization_status, progress_segment in self._progress_segments.items():
            progress_segment.setVisible(True)
            width = full_width if realization_status == ENSEMBLE_STATE_FAILED else 0
            progress_segment.setFixedWidth(width)

    def update_progress(self, status: dict[str, int], realization_count: int) -> None:
        self._status_counts = status
        self._realization_count = realization_count
        if status.get("Finished", 0) < self._realization_count:
            self.start_waiting_progress_bar()
        else:
            self.stop_waiting_progress_bar()
        self.repaint_components()

    @override
    def resizeEvent(self, a0: QResizeEvent | None) -> None:
        self.repaint_components()
