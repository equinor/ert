from __future__ import annotations

from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QMainWindow,
)

from ert.plugins import ErtPluginManager


class EverestMainWindow(QMainWindow):
    def __init__(
        self,
        window_title: str,
    ):
        QMainWindow.__init__(self)

        self.setWindowTitle(window_title)
        self.plugin_manager = ErtPluginManager()
        self.central_widget = QFrame(self)
        self.central_layout = QHBoxLayout(self.central_widget)
        self.central_layout.setContentsMargins(0, 0, 0, 0)
        self.central_layout.setSpacing(0)
        self.central_widget.setLayout(self.central_layout)

        self.central_widget.setMinimumWidth(1500)
        self.central_widget.setMinimumHeight(800)
        self.setCentralWidget(self.central_widget)
