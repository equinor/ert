from typing import Optional

from qtpy.QtCore import Qt, Signal
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
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowCloseButtonHint)

        layout = QVBoxLayout()
        layout.setSizeConstraint(QLayout.SetFixedSize)  # not resizable!!!
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

    def keyPressEvent(self, q_key_event):
        if self.close_button.isEnabled() or q_key_event.key() != Qt.Key_Escape:
            QDialog.keyPressEvent(self, q_key_event)
