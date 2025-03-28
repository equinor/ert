from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog, QHBoxLayout, QPushButton, QVBoxLayout, QWidget

if TYPE_CHECKING:
    from PyQt6.QtGui import QKeyEvent


class ClosableDialog(QDialog):
    def __init__(
        self, title: str | None, widget: QWidget, parent: QWidget | None = None
    ) -> None:
        QDialog.__init__(self, parent)
        self.setWindowTitle(title or "")
        self.setModal(True)
        self.setWindowFlag(Qt.WindowType.CustomizeWindowHint, True)
        self.setWindowFlag(Qt.WindowType.WindowContextHelpButtonHint, False)
        self.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)

        layout = QVBoxLayout()
        layout.addWidget(widget, stretch=1)

        self.__button_layout = QHBoxLayout()
        self.close_button = QPushButton("Close")
        self.close_button.setAutoDefault(False)
        self.close_button.setObjectName("CLOSE")
        self.close_button.clicked.connect(self.accept)
        self.__button_layout.addStretch()
        self.__button_layout.addWidget(self.close_button)

        layout.addLayout(self.__button_layout)

        self.setLayout(layout)

    def disableCloseButton(self) -> None:
        self.close_button.setEnabled(False)

    def enableCloseButton(self) -> None:
        self.close_button.setEnabled(True)

    def keyPressEvent(self, a0: QKeyEvent | None) -> None:
        if self.close_button.isEnabled() or a0 is None or a0.key() != Qt.Key.Key_Escape:
            QDialog.keyPressEvent(self, a0)

    def addButton(self, caption: str, listener: Callable[..., None]) -> QPushButton:
        button = QPushButton(caption)
        button.setObjectName(str(caption).capitalize())
        self.__button_layout.insertWidget(1, button)
        button.clicked.connect(listener)
        return button

    def toggleButton(self, caption: str, enabled: bool) -> None:
        button = self.findChild(QPushButton, str(caption).capitalize())
        if button is not None:
            button.setEnabled(enabled)
