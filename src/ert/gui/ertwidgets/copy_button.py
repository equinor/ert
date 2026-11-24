from abc import abstractmethod

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication, QMessageBox, QPushButton, QSizePolicy

from ert.gui.detect_mode import is_dark_mode
from ert.gui.icon_utils import load_icon, load_stateful_icon

# EDS interactive colours. Not imported from the suggestor module, which
# imports this module.
_PRIMARY_RESTING = "#007079"
_PRIMARY_HOVER = "#004f55"
_PRIMARY_HOVER_ALT = "#deedee"
_DISABLED_BORDER = "#dcdcdc"
_DISABLED_TEXT = "#bebebe"

# EDS outlined primary button
_OUTLINED_PRIMARY_STYLE = f"""
QPushButton {{
    background-color: transparent;
    border: 1px solid {_PRIMARY_RESTING};
    border-radius: 4px;
    padding: 4px 12px;
}}
QPushButton:hover {{
    background-color: {_PRIMARY_HOVER_ALT};
    border-color: {_PRIMARY_HOVER};
}}
QPushButton:disabled {{
    background-color: transparent;
    border-color: {_DISABLED_BORDER};
}}
"""


def _load_button_icon(name: str) -> QIcon:
    if is_dark_mode():
        return load_icon(name)
    return load_stateful_icon(
        name,
        resting=_PRIMARY_RESTING,
        active=_PRIMARY_RESTING,
        disabled=_DISABLED_TEXT,
    )


class CopyButton(QPushButton):
    def __init__(self) -> None:
        super().__init__()
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        if not is_dark_mode():
            self.setStyleSheet(_OUTLINED_PRIMARY_STYLE)
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
