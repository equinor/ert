from __future__ import annotations

from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtWidgets import QTabWidget, QVBoxLayout, QWidget


class TabGroupWidget(QWidget):
    currentTabChanged = Signal()

    def __init__(self, iteration: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.iteration = iteration
        self._tabs = QTabWidget(self)
        self._tabs.currentChanged.connect(lambda _index: self.currentTabChanged.emit())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._tabs)

    @property
    def _tab_widget(self) -> QTabWidget:
        return self._tabs

    def current_widget(self) -> QWidget | None:
        return self._tabs.currentWidget()
