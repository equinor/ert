from typing import Optional

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QKeyEvent
from qtpy.QtWidgets import (
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
        self, title: str, widget: QWidget, parent: Optional[QWidget] = None
    ) -> None:
        QDialog.__init__(self, parent)

        self.setWindowTitle(title)
        self.setModal(True)
        self.setWindowFlags(
            self.windowFlags()
            & ~Qt.WindowFlags(Qt.WindowType.WindowContextHelpButtonHint)
        )
        self.setWindowFlags(
            self.windowFlags() & ~Qt.WindowFlags(Qt.WindowType.WindowCloseButtonHint)
        )

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

    def keyPressEvent(self, a0: Optional[QKeyEvent]) -> None:
        # disallow pressing escape to close
        # when close button is not enabled
        if (
            self._close_button.isEnabled()
            or a0 is None
            or a0.key() != Qt.Key.Key_Escape
        ):
            QDialog.keyPressEvent(self, a0)
