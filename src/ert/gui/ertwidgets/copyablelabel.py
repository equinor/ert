from typing import override

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel

from .copy_button import CopyButton

_ZERO_WIDTH_SPACE = "\u200b"


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
    def __init__(self, label: QLabel, *, wrap: bool) -> None:
        super().__init__()
        self.label = label
        self.wrap = wrap

    @override
    def copy(self) -> None:
        text = unescape_string(self.label.text())
        if self.wrap:
            text = text.replace("/" + _ZERO_WIDTH_SPACE, "/")
        self.copy_text(strip_runpath_magic_keywords(text))


class CopyableLabel(QHBoxLayout):
    """CopyableLabel shows a string that is copyable via
    selection or clicking of a copy button

    With ``wrap`` a long path is broken into several lines after its slashes.
    """

    def __init__(self, text: str, *, wrap: bool = False) -> None:
        super().__init__()

        displayed = escape_string(text)
        if wrap:
            displayed = displayed.replace("/", "/" + _ZERO_WIDTH_SPACE)
        self.label = QLabel(f"<b>{displayed}</b>")
        self.label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.label.setWordWrap(wrap)

        self.copy_button = _CopyButton(self.label, wrap=wrap)

        if wrap:
            self.addWidget(self.label, 1)
            self.addWidget(self.copy_button, alignment=Qt.AlignmentFlag.AlignTop)
        else:
            self.addWidget(self.label)
            self.addWidget(self.copy_button, alignment=Qt.AlignmentFlag.AlignLeft)
            self.setStretch(1, 1)
