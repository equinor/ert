from typing import cast

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtGui import QCloseEvent, QKeyEvent, QMovie
from PyQt6.QtWidgets import (
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLayout,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)


class ProcessJobDialog(QDialog):
    disposeDialog = Signal()
    presentInformation = Signal(str, str, str)
    presentError = Signal(str, str, str)

    closeButtonPressed = Signal()
    cancelConfirmed = Signal()

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        QDialog.__init__(self, parent)

        self.__parent = parent
        self.setWindowTitle(title)
        self.setModal(True)
        self.setWindowFlag(Qt.WindowType.WindowContextHelpButtonHint, False)
        self.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)

        layout = QVBoxLayout()
        layout.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)

        widget = QWidget()
        widget_layout = QHBoxLayout()

        size = 64
        spin_movie = QMovie("img:loading.gif")
        spin_movie.setSpeed(60)
        spin_movie.setScaledSize(QSize(size, size))
        spin_movie.start()

        processing_animation = QLabel()
        processing_animation.setMaximumSize(QSize(size, size))
        processing_animation.setMinimumSize(QSize(size, size))
        processing_animation.setMovie(spin_movie)
        widget_layout.addWidget(processing_animation)

        self.processing_label = QLabel(f"Processing job: '{title}'")
        widget_layout.addWidget(self.processing_label, Qt.AlignmentFlag.AlignBottom)

        widget.setLayout(widget_layout)

        layout.addWidget(widget)

        button_layout = QHBoxLayout()
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.closeButtonPressed.emit)
        button_layout.addStretch()
        button_layout.addWidget(self.close_button)

        layout.addStretch()
        layout.addLayout(button_layout)

        self.setLayout(layout)

        self.disposeDialog.connect(self.reject)
        self.presentInformation.connect(self.__presentInformation)
        self.presentError.connect(self.__presentError)
        self.closeButtonPressed.connect(self.__confirmCancel)

        self._msg_box: QMessageBox | None = None

    def disableCloseButton(self) -> None:
        self.close_button.setEnabled(False)

    def enableCloseButton(self) -> None:
        self.close_button.setEnabled(True)

    def keyPressEvent(self, a0: QKeyEvent | None) -> None:
        # disallow pressing escape to close
        # when close button is not enabled
        if self.close_button.isEnabled() or a0 is None or a0.key() != Qt.Key.Key_Escape:
            QDialog.keyPressEvent(self, a0)

    def closeEvent(self, a0: QCloseEvent | None) -> None:
        if a0 is not None:
            a0.ignore()
        self.closeButtonPressed.emit()

    def __createMsgBox(self, title: str, message: str, details: str) -> QMessageBox:
        msg_box = QMessageBox(cast(QWidget | None, self.parent()))
        msg_box.setText(title)
        msg_box.setInformativeText(message)

        if len(details) > 0:
            msg_box.setDetailedText(details)

        horizontal_spacer = QSpacerItem(
            500, 0, QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Expanding
        )
        layout = cast(QGridLayout, msg_box.layout())
        layout.addItem(horizontal_spacer, layout.rowCount(), 0, 1, layout.columnCount())

        return msg_box

    def __presentInformation(self, title: str, message: str, details: str) -> None:
        self._msg_box = self.__createMsgBox(title, message, details)
        self._msg_box.setIcon(QMessageBox.Icon.Information)

        self._msg_box.exec()

    def __presentError(self, title: str, message: str, details: str) -> None:
        self._msg_box = self.__createMsgBox(title, message, details)
        self._msg_box.setIcon(QMessageBox.Icon.Critical)

        self._msg_box.exec()

    def __confirmCancel(self) -> None:
        cancel_box = self.__createMsgBox(
            "Confirm cancel", "Are you sure you want to cancel the running job?", ""
        )
        cancel_box.setIcon(QMessageBox.Icon.Question)
        cancel_box.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        cancel_box.exec()

        cancel = cancel_box.result()

        if cancel == QMessageBox.StandardButton.Yes:
            self.cancelConfirmed.emit()
