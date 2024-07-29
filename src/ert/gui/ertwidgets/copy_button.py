from abc import abstractmethod

from qtpy.QtCore import QTimer
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import (
    QApplication,
    QMessageBox,
    QPushButton,
    QSizePolicy,
)


class CopyButton(QPushButton):
    def __init__(self) -> None:
        super().__init__()
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.setIcon(QIcon("img:copy.svg"))
        self.restore_timer = QTimer(self)

        def restore_text() -> None:
            self.setIcon(QIcon("img:copy.svg"))

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
                "Cannot copy text to clipboard because your system does not have a clipboard",
                QMessageBox.Ok,
            )
        self.setIcon(QIcon("img:check.svg"))
        self.restore_timer.start(1000)
