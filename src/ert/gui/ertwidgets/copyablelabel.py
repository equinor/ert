from typing import override

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel

from .copy_button import CopyButton

# Zero-width space, lets a long path wrap after each "/".
_BREAK_OPPORTUNITY = "\u200b"


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
        string.replace(_BREAK_OPPORTUNITY, "")
        .replace("&amp;", "&")
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


class CopyableLabel(QHBoxLayout):
    """CopyableLabel shows a string that is copyable via
    selection or clicking of a copy button
    """

    def __init__(self, text: str, *, wrap: bool = False) -> None:
        super().__init__()

        escaped = escape_string(text)
        if wrap:
            escaped = escaped.replace("/", "/" + _BREAK_OPPORTUNITY)
        self.label = QLabel(f"<b>{escaped}</b>")
        if wrap:
            self.label.setWordWrap(True)
            # Selecting wrapped text would include the zero-width spaces, so
            # copying is left to the copy button.
            self.label.setToolTip(text)
        else:
            self.label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )

        self.copy_button = _CopyButton(self.label)

        self.addWidget(self.label)
        self.addWidget(
            self.copy_button,
            alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
            if wrap
            else Qt.AlignmentFlag.AlignLeft,
        )

        # A wrapping label needs the width, otherwise it wraps more than needed.
        self.setStretch(0 if wrap else 1, 1)
