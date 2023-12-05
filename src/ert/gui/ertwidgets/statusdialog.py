from typing import List, cast

from qtpy.QtCore import QObject, Qt, Signal, Slot
from qtpy.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QPushButton,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from ert.run_models import RunModelEvent, RunModelStatusEvent, RunModelTimeEvent


class StatusDialog(QDialog):
    close = Signal()
    run = Signal()

    def __init__(self, title: str, widget: QWidget, parent: QObject = None):
        QDialog.__init__(self, parent)

        self.setWindowTitle(title)
        self.setModal(True)
        self.setWindowFlags(self.windowFlags() | Qt.CustomizeWindowHint)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowCloseButtonHint)

        layout = QVBoxLayout()
        layout.addWidget(widget)

        run_button = QPushButton("Run")
        run_button.setAutoDefault(True)
        run_button.setObjectName("RUN")
        run_button.clicked.connect(self.run)

        self._close_button = QPushButton("Close")
        self._close_button.setAutoDefault(False)
        self._close_button.setObjectName("CLOSE")
        self._close_button.clicked.connect(self.accept)

        self._status_bar = QStatusBar()

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(run_button)
        button_layout.addWidget(self._close_button)

        layout.addWidget(self._status_bar)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def keyPressEvent(self, q_key_event):
        if not self._close_button.isEnabled() and q_key_event.key() == Qt.Key_Escape:
            pass
        else:
            QDialog.keyPressEvent(self, q_key_event)

    def enable_button(self, caption, enabled: bool = True):
        button = cast(
            QPushButton, self.findChild(QPushButton, str(caption).capitalize())
        )
        if button is not None:
            button.setEnabled(enabled)

    def enable_buttons(self, enabled: bool = True):
        buttons = cast(List[QPushButton], self.findChildren(QPushButton))
        for button in buttons:
            button.setEnabled(enabled)

    @Slot(RunModelEvent)
    def progress_update(self, event: RunModelEvent):
        if isinstance(event, RunModelStatusEvent):
            self._status_bar.showMessage(f"{event.msg}")
        elif isinstance(event, RunModelTimeEvent):
            self._status_bar.showMessage(
                f"Estimated remaining time {event.remaining_time:.2f}s"
            )

    @Slot()
    def clear_status(self):
        self._status_bar.clearMessage()
