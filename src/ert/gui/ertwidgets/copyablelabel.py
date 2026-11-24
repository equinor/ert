from typing import override

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QPainter, QPaintEvent
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy

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


class _ElidedLabel(QLabel):
    """Bold single line label that elides its text in the middle to fit."""

    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.setTextFormat(Qt.TextFormat.PlainText)
        font = self.font()
        font.setBold(True)
        self.setFont(font)
        self.setToolTip(text)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

    @override
    def minimumSizeHint(self) -> QSize:
        return QSize(0, super().minimumSizeHint().height())

    @override
    def paintEvent(self, a0: QPaintEvent | None) -> None:
        painter = QPainter(self)
        rect = self.contentsRect()
        elided = self.fontMetrics().elidedText(
            self.text(), Qt.TextElideMode.ElideMiddle, rect.width()
        )
        painter.drawText(rect, int(self.alignment()), elided)


class CopyableLabel(QHBoxLayout):
    """CopyableLabel shows a string that is copyable via
    selection or clicking of a copy button
    """

    def __init__(self, text: str, *, elide: bool = False) -> None:
        super().__init__()

        if elide:
            # The full text is in the tooltip and copied by the copy button.
            self.label: QLabel = _ElidedLabel(text)
        else:
            self.label = QLabel(f"<b>{escape_string(text)}</b>")
            self.label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )

        self.copy_button = _CopyButton(self.label)

        self.addWidget(self.label, 1)
        self.addWidget(self.copy_button, alignment=Qt.AlignmentFlag.AlignLeft)
