from typing import Any, List, Optional, cast

from qtpy.QtCore import Qt, Signal, Slot
from qtpy.QtGui import QKeyEvent
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
    close = Signal()  # type: ignore
    run = Signal()

    def __init__(
        self, title: str, widget: QWidget, parent: Optional[QWidget] = None
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

    def keyPressEvent(self, a0: Optional[QKeyEvent]) -> None:
        # disallow pressing escape to close
        # when close button is not enabled
        if (
            self._close_button.isEnabled()
            or a0 is None
            or a0.key() != Qt.Key.Key_Escape
        ):
            QDialog.keyPressEvent(self, a0)

    def enable_button(self, caption: Any, enabled: bool = True) -> None:
        button = self.findChild(QPushButton, str(caption).capitalize())
        if button is not None:
            button.setEnabled(enabled)

    def enable_buttons(self, enabled: bool = True) -> None:
        buttons = cast(List[QPushButton], self.findChildren(QPushButton))
        for button in buttons:
            button.setEnabled(enabled)

    @Slot(RunModelEvent)
    def progress_update(self, event: RunModelEvent) -> None:
        if isinstance(event, RunModelStatusEvent):
            self._status_bar.showMessage(f"{event.msg}")
        elif isinstance(event, RunModelTimeEvent):
            self._status_bar.showMessage(
                f"Estimated remaining time {event.remaining_time:.2f}s"
            )

    @Slot()
    def clear_status(self) -> None:
        self._status_bar.clearMessage()
