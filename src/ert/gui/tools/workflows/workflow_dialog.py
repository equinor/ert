from PyQt6.QtCore import Qt
from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class WorkflowDialog(QDialog):
    closeButtonPressed = Signal()

    def __init__(
        self, title: str, widget: QWidget, parent: QWidget | None = None
    ) -> None:
        QDialog.__init__(self, parent)

        self.setWindowTitle(title)
        self.setModal(True)
        self.setWindowFlag(Qt.WindowType.WindowContextHelpButtonHint, False)
        self.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)

        layout = QVBoxLayout()
        layout.setSizeConstraint(
            QLayout.SizeConstraint.SetFixedSize
        )  # not resizable!!!
        layout.addWidget(widget)

        button_layout = QHBoxLayout()
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.closeButtonPressed.emit)
        button_layout.addStretch()
        button_layout.addWidget(self.close_button)

        layout.addStretch()
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def disableCloseButton(self) -> None:
        self.close_button.setEnabled(False)

    def enableCloseButton(self) -> None:
        self.close_button.setEnabled(True)

    def keyPressEvent(self, a0: QKeyEvent | None) -> None:
        # disallow pressing escape to close
        # when close button is not enabled
        if self.close_button.isEnabled() or a0 is None or a0.key() != Qt.Key.Key_Escape:
            QDialog.keyPressEvent(self, a0)
