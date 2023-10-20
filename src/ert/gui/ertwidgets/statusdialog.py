from typing import List, cast

from qtpy.QtCore import QObject, Qt, Signal, Slot
from qtpy.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLayout,
    QPushButton,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from ert.analysis import AnalysisEvent


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
        layout.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)
        layout.addWidget(widget)

        self.run_button = QPushButton("Run")
        self.run_button.setAutoDefault(True)
        self.run_button.setObjectName("RUN")
        self.run_button.clicked.connect(self.run)

        self.close_button = QPushButton("Close")
        self.close_button.setAutoDefault(False)
        self.close_button.setObjectName("CLOSE")
        self.close_button.clicked.connect(self.accept)

        self.status_bar = QStatusBar()

        self.__button_layout = QHBoxLayout()
        self.__button_layout.addWidget(self.status_bar)
        self.__button_layout.addWidget(self.run_button)
        self.__button_layout.addWidget(self.close_button)

        layout.addStretch()
        layout.addLayout(self.__button_layout)

        self.setLayout(layout)

    def keyPressEvent(self, q_key_event):
        if not self.close_button.isEnabled() and q_key_event.key() == Qt.Key_Escape:
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

    @Slot(object)
    def progress_update(self, event: AnalysisEvent):
        self.status_bar.showMessage(f"{event.msg}")

    @Slot()
    def clear_status(self):
        self.status_bar.clearMessage()
