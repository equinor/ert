from abc import abstractmethod

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication, QMessageBox, QPushButton, QSizePolicy

from ert.gui.detect_mode import is_dark_mode
from ert.gui.eds_colors import DARK_GREY, MEDIUM_GREEN
from ert.gui.icon_utils import load_icon, load_stateful_icon


def _load_button_icon(name: str) -> QIcon:
    if is_dark_mode():
        return load_icon(name)
    return load_stateful_icon(
        name,
        resting=MEDIUM_GREEN,
        active=MEDIUM_GREEN,
        disabled=DARK_GREY,
    )


class CopyButton(QPushButton):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("copy_button")
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.setIcon(_load_button_icon("copy.svg"))
        self.restore_timer = QTimer(self)

        def restore_text() -> None:
            self.setIcon(_load_button_icon("copy.svg"))

        self.restore_timer.timeout.connect(restore_text)

        self.clicked.connect(self.copy)

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
        self.setIcon(_load_button_icon("check.svg"))
        self.restore_timer.start(1000)
