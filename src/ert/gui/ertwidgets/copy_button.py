from abc import abstractmethod
from typing import override

from PyQt6.QtCore import QEvent, QTimer
from PyQt6.QtGui import QPalette
from PyQt6.QtWidgets import QApplication, QMessageBox, QPushButton, QSizePolicy

from ert.gui.icon_utils import load_stateful_icon


class CopyButton(QPushButton):
    """Button whose icon has the text color given by the style sheet."""

    _icon_name = "copy.svg"

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("copy_button")
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self._show_icon("copy.svg")
        self.restore_timer = QTimer(self)
        self.restore_timer.timeout.connect(lambda: self._show_icon("copy.svg"))

        self.clicked.connect(self.copy)

    def _show_icon(self, name: str) -> None:
        self._icon_name = name
        palette = self.palette()
        role = self.foregroundRole()
        color = palette.color(QPalette.ColorGroup.Normal, role).name()
        disabled_color = palette.color(QPalette.ColorGroup.Disabled, role).name()
        self.setIcon(
            load_stateful_icon(
                name, resting=color, active=color, disabled=disabled_color
            )
        )

    @override
    def changeEvent(self, event: QEvent | None) -> None:
        super().changeEvent(event)
        if event is not None and event.type() == QEvent.Type.PaletteChange:
            self._show_icon(self._icon_name)

    @abstractmethod
    def copy(self) -> None:
        pass

    def copy_text(self, text: str) -> None:
        clipboard = QApplication.clipboard()
        if clipboard:
            clipboard.setText(text)
        else:
            QMessageBox.critical(
                None,
                "Error",
                "Cannot copy text to clipboard because your "
                "system does not have a clipboard",
                QMessageBox.StandardButton.Ok,
            )
        self._show_icon("check.svg")
        self.restore_timer.start(1000)
