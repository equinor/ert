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

from ._colors import (
    BLUE_BACKGROUND,
    BLUE_TEXT,
    RED_BACKGROUND,
    RED_TEXT,
    YELLOW_BACKGROUND,
    YELLOW_TEXT,
)


def _svg_icon(image_name):
    widget = QtSvg.QSvgWidget(f"img:{image_name}.svg")
    widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    return widget


class SuggestorMessage(QWidget):
    def __init__(self, header, text_color, bg_color, icon, info):
        super().__init__()
        self.setAttribute(Qt.WA_StyledBackground)
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
        info.message = info.message.replace("<", "&lt;").replace(">", "&gt;")
        self.lbl = QLabel(
            '<div style="font-size: 16px; line-height: 24px;">'
            + f'<b style="color: {text_color}">'
            + header
            + "</b>"
            + info.message
            + "<p>"
            + info.location()
            + "</p>"
            + "</div>"
        )
        self.lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lbl.setWordWrap(True)

        self.hbox = QHBoxLayout()
        self.hbox.setContentsMargins(16, 16, 16, 16)
        self.hbox.addWidget(self.icon, alignment=Qt.AlignTop)
        self.hbox.addWidget(self.lbl, alignment=Qt.AlignTop)
        self.setLayout(self.hbox)

    @classmethod
    def error_msg(cls, info):
        return SuggestorMessage(
            "Error: ", RED_TEXT, RED_BACKGROUND, _svg_icon("error"), info
        )

    @classmethod
    def warning_msg(cls, info):
        return SuggestorMessage(
            "Warning: ", YELLOW_TEXT, YELLOW_BACKGROUND, _svg_icon("warning"), info
        )

    @classmethod
    def deprecation_msg(cls, info):
        return SuggestorMessage(
            "Deprecation: ", BLUE_TEXT, BLUE_BACKGROUND, _svg_icon("bell"), info
        )
