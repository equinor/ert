from typing import override

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QResizeEvent
from PyQt6.QtWidgets import QHBoxLayout, QLabel

from .copy_button import CopyButton


def escape_string(string: str) -> str:
    """
    Designed to replace/escape invalid html characters for
    correct display in Qt QWidgets

    >>> escape_string("realization-<IENS>/iteration-<ITER>")
    'realization-&lt;IENS&gt;/iteration-&lt;ITER&gt;'

    >>> escape_string("<some>&<thing>")
    '&lt;some&gt;&amp;&lt;thing&gt;'
    """
    return string.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def unescape_string(string: str) -> str:
    """
    Designed to transform an escaped string within a Qt widget
    into an unescaped string.

    >>> unescape_string("<b>realization-&lt;IENS&gt;/iteration-&lt;ITER&gt;</b>")
    'realization-<IENS>/iteration-<ITER>'

    >>> unescape_string("<b>&lt;some&gt;&amp;&lt;thing&gt;</b>")
    '<some>&<thing>'
    """
    return (
        string.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("<b>", "")
        .replace("</b>", "")
    )


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
        self.copy_text(strip_runpath_magic_keywords(unescape_string(self.label.text())))


class _WrappedLabel(QLabel):
    """A word-wrapped label that reserves the height its lines need."""

    @override
    def resizeEvent(self, event: QResizeEvent | None) -> None:
        super().resizeEvent(event)
        # heightForWidth() never reports less than the current minimum height
        self.setMinimumHeight(0)
        self.setMinimumHeight(self.heightForWidth(self.width()))


class CopyableLabel(QHBoxLayout):
    """CopyableLabel shows a string that is copyable via
    selection or clicking of a copy button

    With ``wrap`` a long path is broken into several lines after its slashes.
    """

    def __init__(self, text: str, *, wrap: bool = False) -> None:
        super().__init__()

        label_class = _WrappedLabel if wrap else QLabel
        self.label = label_class(f"<b>{escape_string(text)}</b>")
        self.label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.label.setWordWrap(wrap)

        self.copy_button = _CopyButton(self.label)

        if wrap:
            self.addWidget(self.label, 1)
            self.addWidget(self.copy_button, alignment=Qt.AlignmentFlag.AlignTop)
        else:
            self.addWidget(self.label)
            self.addWidget(self.copy_button, alignment=Qt.AlignmentFlag.AlignLeft)
            self.setStretch(1, 1)
