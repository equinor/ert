from __future__ import annotations

import html
from typing import TYPE_CHECKING, Optional

from qtpy.QtCore import (
    QObject,
    QPoint,
    Qt,
    Signal,
)
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QFrame,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from qtpy.QtCore import QEvent
    from qtpy.QtGui import QHideEvent


class ErrorPopup(QWidget):
    error_template = (
        "<html>"
        "<table style='background-color: #ffdfdf;'width='100%%'>"
        "<tr><td style='font-weight: bold; padding-left: 5px;'>Warning:</td></tr>"
        "%s"
        "</table>"
        "</html>"
    )

    def __init__(self) -> None:
        QWidget.__init__(self, None, Qt.WindowType.ToolTip)
        self.resize(300, 50)

        self.setContentsMargins(0, 0, 0, 0)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        self._error_widget = QLabel("")
        self._error_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        self._error_widget.setFrameStyle(QFrame.Box)
        self._error_widget.setWordWrap(True)
        self._error_widget.setScaledContents(True)
        self._error_widget.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self._error_widget)

        self.setLayout(layout)

    def presentError(self, widget: QWidget, error: str) -> None:
        assert isinstance(widget, QWidget)

        self._error_widget.setText(ErrorPopup.error_template % html.escape(error))
        self.show()

        size_hint = self.sizeHint()
        rect = widget.rect()
        p = widget.mapToGlobal(QPoint(rect.left(), rect.top()))

        self.setGeometry(
            p.x(), p.y() - size_hint.height() - 5, size_hint.width(), size_hint.height()
        )

        self.raise_()


class ValidationSupport(QObject):
    STRONG_ERROR_COLOR = QColor(255, 215, 215)
    ERROR_COLOR = QColor(255, 235, 235)
    INVALID_COLOR = QColor(235, 235, 255)

    WARNING = "warning"
    EXCLAMATION = "ide/small/exclamation"

    validationChanged = Signal(bool)

    def __init__(self, validation_target: QWidget) -> None:
        QObject.__init__(self)

        self._validation_target = validation_target
        self._validation_message: Optional[str] = None
        self._validation_type: Optional[str] = None
        self._error_popup = ErrorPopup()

        self._originalEnterEvent = validation_target.enterEvent
        self._originalLeaveEvent = validation_target.leaveEvent
        self._originalHideEvent = validation_target.hideEvent

        def enterEvent(a0: Optional[QEvent]) -> None:
            self._originalEnterEvent(a0)

            if not self.isValid():
                self._error_popup.presentError(
                    self._validation_target,
                    self._validation_message if self._validation_message else "",
                )

        validation_target.enterEvent = enterEvent  # type: ignore[method-assign]

        def leaveEvent(a0: Optional[QEvent]) -> None:
            self._originalLeaveEvent(a0)

            if self._error_popup is not None:
                self._error_popup.hide()

        validation_target.leaveEvent = leaveEvent  # type: ignore[method-assign]

        def hideEvent(a0: Optional[QHideEvent]) -> None:
            self._error_popup.hide()
            self._originalHideEvent(a0)

        validation_target.hideEvent = hideEvent  # type: ignore[method-assign]

    def setValidationMessage(
        self, message: str, validation_type: str = WARNING
    ) -> None:
        """Add a warning or information icon to the widget with a tooltip"""
        message = message.strip()
        if not message:
            self._validation_type = None
            self._validation_message = None
            self._error_popup.hide()
            self.validationChanged.emit(True)

        else:
            self._validation_type = validation_type
            self._validation_message = message
            if (
                self._validation_target.hasFocus()
                or self._validation_target.underMouse()
            ):
                self._error_popup.presentError(
                    self._validation_target, self._validation_message
                )
            self.validationChanged.emit(False)

    def isValid(self) -> bool:
        return self._validation_message is None
