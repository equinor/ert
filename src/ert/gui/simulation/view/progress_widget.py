from __future__ import annotations

from PyQt6.QtGui import QColor, QResizeEvent
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QVBoxLayout,
)

from ert.ensemble_evaluator.state import ENSEMBLE_STATE_FAILED, REAL_STATE_TO_COLOR


class ProgressWidget(QFrame):
    def __init__(self) -> None:
        super().__init__()
        self.setFixedHeight(70)

        self._vertical_layout = QVBoxLayout(self)
        self._vertical_layout.setContentsMargins(0, 0, 0, 0)
        self._vertical_layout.setSpacing(0)
        self.setLayout(self._vertical_layout)

        self._waiting_progress_bar = QProgressBar(self)
        self._waiting_progress_bar.setRange(0, 0)
        self._waiting_progress_bar.setFixedHeight(30)
        self._vertical_layout.addWidget(self._waiting_progress_bar)

        self._progress_frame = QFrame(self)
        self._vertical_layout.addWidget(self._progress_frame)

        self._horizontal_layout = QHBoxLayout(self._progress_frame)
        self._horizontal_layout.setContentsMargins(0, 0, 0, 0)
        self._horizontal_layout.setSpacing(0)
        self._progress_frame.setLayout(self._horizontal_layout)

        self._legend_frame = QFrame(self)
        self._vertical_layout.addWidget(self._legend_frame)
        self._legend_frame.setFixedHeight(30)
        self._horizontal_legend_layout = QHBoxLayout(self._legend_frame)
        self._horizontal_legend_layout.setContentsMargins(0, 0, 0, 0)
        self._horizontal_legend_layout.setSpacing(0)

        self._status: dict[str, int] = {}
        self._realization_count = 0
        self._progress_label_map: dict[str, QLabel] = {}
        self._legend_map_text = {}

        for state, color in REAL_STATE_TO_COLOR.items():
            label = QLabel(self)
            label.setVisible(False)
            label.setObjectName(f"progress_{state}")
            label.setStyleSheet(f"background-color : {QColor(*color).name()}")
            self._progress_label_map[state] = label
            self._horizontal_layout.addWidget(label)

            label = QLabel(self)
            label.setFixedSize(20, 20)
            label.setStyleSheet(
                f"background-color : {QColor(*color).name()}; border: 1px solid black;"
            )
            self._horizontal_legend_layout.addWidget(label)

            label = QLabel(self)
            label.setObjectName(f"progress_label_text_{state}")
            label.setText(f" {state} ({0}/{0})")
            self._legend_map_text[state] = label
            self._horizontal_legend_layout.addWidget(label)

    def repaint_components(self) -> None:
        if self._realization_count > 0:
            full_width = self.width()
            self.stop_waiting_progress_bar()

            for state, label in self._progress_label_map.items():
                label.setVisible(True)
                count = self._status.get(state, 0)
                width = int((count / self._realization_count) * full_width)
                label.setFixedWidth(width)

            for state, label in self._legend_map_text.items():
                label.setText(
                    f" {state} ({self._status.get(state, 0)}/{self._realization_count})"
                )

    def stop_waiting_progress_bar(self) -> None:
        self._waiting_progress_bar.setVisible(False)

    def start_waiting_progress_bar(self) -> None:
        self._waiting_progress_bar.setVisible(True)

    def set_all_failed(self) -> None:
        self.stop_waiting_progress_bar()
        full_width = self.width()
        for state, label in self._progress_label_map.items():
            label.setVisible(True)
            width = full_width if state == ENSEMBLE_STATE_FAILED else 0
            label.setFixedWidth(width)

    def update_progress(self, status: dict[str, int], realization_count: int) -> None:
        self._status = status
        self._realization_count = realization_count
        if status.get("Finished", 0) < self._realization_count:
            self.start_waiting_progress_bar()
        else:
            self.stop_waiting_progress_bar()
        self.repaint_components()

    def resizeEvent(self, a0: QResizeEvent | None) -> None:
        self.repaint_components()
