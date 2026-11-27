from typing import override

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel

from .copy_button import CopyButton


def strip_runpath_magic_keywords(runpath: str) -> str:
    rp_stripped = ""
    for s in runpath.split("/"):
        if all(substring not in s for substring in ("<IENS>", "<ITER>")) and s:
            rp_stripped += "/" + s
    if not rp_stripped:
        rp_stripped = "/"

    if runpath and not runpath.startswith("/"):
        rp_stripped = rp_stripped[1:]

    return rp_stripped


class _CopyButton(CopyButton):
    def __init__(self, label: QLabel) -> None:
        super().__init__()
        self.label = label

    @override
    def copy(self) -> None:
        self.copy_text(strip_runpath_magic_keywords(self.label.text()))


class CopyableLabel(QHBoxLayout):
    """CopyableLabel shows a string that is copyable via
    selection or clicking of a copy button
    """

    def __init__(self, text: str) -> None:
        super().__init__()

        self.label = QLabel(text)
        self.label.setTextFormat(Qt.TextFormat.PlainText)
        font = self.label.font()
        font.setBold(True)
        self.label.setFont(font)
        self.label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.copy_button = _CopyButton(self.label)

        self.addWidget(self.label)
        self.addWidget(self.copy_button, alignment=Qt.AlignmentFlag.AlignLeft)

        self.setStretch(1, 1)
