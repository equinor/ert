from __future__ import annotations

from PyQt5 import QtSvg
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QWidget,
)
from typing_extensions import Self

from ._colors import (
    BLUE_BACKGROUND,
    BLUE_TEXT,
    RED_BACKGROUND,
    RED_TEXT,
    YELLOW_BACKGROUND,
    YELLOW_TEXT,
)


def _svg_icon(image_name: str) -> QtSvg.QSvgWidget:
    widget = QtSvg.QSvgWidget(f"img:{image_name}.svg")
    widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    return widget


class SuggestorMessage(QWidget):
    def __init__(
        self,
        header: str,
        text_color: str,
        bg_color: str,
        icon: QWidget,
        message: str,
        location: str,
    ) -> None:
        super().__init__()
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.setStyleSheet(
            f"""
            background-color: {bg_color};
            border-radius: 4px;
        """
        )
        shadowEffect = QGraphicsDropShadowEffect()
        shadowEffect.setColor(QColor.fromRgb(0, 0, 0, int(0.12 * 255)))
        shadowEffect.setBlurRadius(5)
        shadowEffect.setXOffset(2)
        shadowEffect.setYOffset(4)
        self.setGraphicsEffect(shadowEffect)
        self.setContentsMargins(0, 0, 0, 0)

        self.icon = icon
        message = message.replace("<", "&lt;").replace(">", "&gt;")
        self.lbl = QLabel(
            '<div style="font-size: 16px; line-height: 24px;">'
            + f'<b style="color: {text_color}">'
            + header
            + "</b>"
            + message
            + "<p>"
            + location
            + "</p>"
            + "</div>"
        )
        self.lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.lbl.setWordWrap(True)

        self.hbox = QHBoxLayout()
        self.hbox.setContentsMargins(16, 16, 16, 16)
        self.hbox.addWidget(self.icon, alignment=Qt.AlignmentFlag.AlignTop)
        self.hbox.addWidget(self.lbl, alignment=Qt.AlignmentFlag.AlignTop)
        self.setLayout(self.hbox)

    @classmethod
    def error_msg(cls, message: str, location: str) -> Self:
        return cls(
            "Error: ", RED_TEXT, RED_BACKGROUND, _svg_icon("error"), message, location
        )

    @classmethod
    def warning_msg(cls, message: str, location: str) -> Self:
        return cls(
            "Warning: ",
            YELLOW_TEXT,
            YELLOW_BACKGROUND,
            _svg_icon("warning"),
            message,
            location,
        )

    @classmethod
    def deprecation_msg(cls, message: str, location: str) -> Self:
        return cls(
            "Deprecation: ",
            BLUE_TEXT,
            BLUE_BACKGROUND,
            _svg_icon("bell"),
            message,
            location,
        )
