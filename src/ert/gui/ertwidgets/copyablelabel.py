from os import path

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
)

from .copy_button import CopyButton

# Get the absolute path of the directory that contains the current script
current_dir = path.dirname(path.abspath(__file__))


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


def strip_run_path_magic_keywords(run_path: str) -> str:
    rp_stripped = ""
    for s in run_path.split("/"):
        if all(substring not in s for substring in ("<IENS>", "<ITER>")) and s:
            rp_stripped += "/" + s
    if not rp_stripped:
        rp_stripped = "/"

    if run_path and not run_path.startswith("/"):
        rp_stripped = rp_stripped[1:]

    return rp_stripped


class _CopyButton(CopyButton):
    def __init__(self, label: QLabel) -> None:
        super().__init__()
        self.label = label

    def copy(self) -> None:
        self.copy_text(
            strip_run_path_magic_keywords(unescape_string(self.label.text()))
        )


class CopyableLabel(QHBoxLayout):
    """CopyableLabel shows a string that is copyable via
    selection or clicking of a copy button"""

    def __init__(self, text: str) -> None:
        super().__init__()

        self.label = QLabel(f"<b>{escape_string(text)}</b>")
        self.label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.copy_button = _CopyButton(self.label)

        self.addWidget(self.label)
        self.addWidget(self.copy_button, alignment=Qt.AlignmentFlag.AlignLeft)
