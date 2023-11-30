from PyQt5 import QtSvg
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QWidget


def _svg_icon(image_name):
    widget = QtSvg.QSvgWidget(f"img:{image_name}.svg")
    widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    widget.setStyleSheet("width: 40px; height: 40px;")
    return widget


class SuggestorMessage(QWidget):
    def __init__(self, header, icon, info):
        super().__init__()
        self.setAttribute(Qt.WA_StyledBackground)
        self.setStyleSheet("background-color: white;")

        self.icon = icon
        info.message = info.message.replace("<", "&lt;").replace(">", "&gt;")
        self.lbl = QLabel(
            "<b>" + header + "</b>" + info.message + "<p>" + info.location() + "</p>"
        )
        self.lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lbl.setWordWrap(True)

        self.hbox = QHBoxLayout()
        self.hbox.setSpacing(16)
        self.hbox.setContentsMargins(16, 0, 16, 0)
        self.hbox.addWidget(self.icon, alignment=Qt.AlignLeft)
        self.hbox.addWidget(self.lbl)
        self.setLayout(self.hbox)

    @classmethod
    def error_msg(cls, info):
        return SuggestorMessage("Error: ", _svg_icon("error_bgcircle"), info)

    @classmethod
    def warning_msg(cls, info):
        return SuggestorMessage("Warning: ", _svg_icon("warning_bgcircle"), info)

    @classmethod
    def deprecation_msg(cls, info):
        return SuggestorMessage("Deprecation: ", _svg_icon("thumbdown_bgcircle"), info)
