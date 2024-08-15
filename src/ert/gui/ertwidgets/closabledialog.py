from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QDialog, QHBoxLayout, QPushButton, QVBoxLayout, QWidget

if TYPE_CHECKING:
    from qtpy.QtGui import QKeyEvent
    from qtpy.QtWidgets import QT_SLOT


class ClosableDialog(QDialog):
    def __init__(
        self, title: Optional[str], widget: QWidget, parent: Optional[QWidget] = None
    ) -> None:
        QDialog.__init__(self, parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.CustomizeWindowHint)
        self.setWindowFlags(
            self.windowFlags()
            & ~Qt.WindowFlags(Qt.WindowType.WindowContextHelpButtonHint)
        )
        self.setWindowFlags(
            self.windowFlags() & ~Qt.WindowFlags(Qt.WindowType.WindowCloseButtonHint)
        )

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

    def keyPressEvent(self, a0: Optional[QKeyEvent]) -> None:
        if self.close_button.isEnabled() or a0 is None or a0.key() != Qt.Key.Key_Escape:
            QDialog.keyPressEvent(self, a0)

    def addButton(self, caption: str, listener: QT_SLOT) -> QPushButton:
        button = QPushButton(caption)
        button.setObjectName(str(caption).capitalize())
        self.__button_layout.insertWidget(1, button)
        button.clicked.connect(listener)
        return button

    def toggleButton(self, caption: str, enabled: bool) -> None:
        button = self.findChild(QPushButton, str(caption).capitalize())
        if button is not None:
            button.setEnabled(enabled)
