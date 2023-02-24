from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QWidget


class SuggestorMessage(QWidget):
    def __init__(self, lbl_text, color, text):
        QWidget.__init__(self)
        common_style = """
        border-style: outset;
        border-width: 2px;
        border-radius: 10px;
        border-color: darkgrey;
        padding: 6px;"""

        self.type_lbl = QLabel(lbl_text)
        self.type_lbl.setAlignment(Qt.AlignCenter)
        self.type_lbl.setStyleSheet(
            common_style
            + f"""
            background-color: {color};
            font: bold 14px;
            max-width: 6em;
            min-width: 6em;
            max-height: 1em;"""
        )

        self.lbl = QLabel(text)
        self.lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lbl.setWordWrap(True)
        self.lbl.setStyleSheet(common_style + "background-color: lightgray;")

        self.hbox = QHBoxLayout()
        self.hbox.addWidget(self.type_lbl)
        self.hbox.addWidget(self.lbl)
        self.setLayout(self.hbox)

    @classmethod
    def error_msg(cls, text):
        color = "#ff0000"
        return SuggestorMessage("ERROR", color, text)

    @classmethod
    def warning_msg(cls, text):
        color = "#ff6a00"
        return SuggestorMessage("WARNING", color, text)

    @classmethod
    def deprecation_msg(cls, text):
        color = "#faf339"
        return SuggestorMessage("DEPRECATION", color, text)
