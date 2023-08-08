from os import path

from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QApplication, QHBoxLayout, QLabel, QPushButton, QSizePolicy

# Get the absolute path of the directory that contains the current script
current_dir = path.dirname(path.abspath(__file__))


def escape_string(string):
    """
    Designed to replace/escape invalid html characters for
    correct display in Qt QWidgets

    >>> escape_string("realization-<IENS>/iteration-<ITER>")
    'realization-&lt;IENS&gt;/iteration-&lt;ITER&gt;'

    >>> escape_string("<some>&<thing>")
    '&lt;some&gt;&amp;&lt;thing&gt;'
    """
    return string.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def unescape_string(string):
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


class CopyableLabel(QHBoxLayout):
    """CopyableLabel shows a string that is copyable via
    selection or clicking of a copy button"""

    def __init__(self, text) -> None:
        super().__init__()

        self.label = QLabel(f"<b>{escape_string(text)}</b>")
        self.label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.copy_button = QPushButton("")
        self.copy_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        icon_path = path.join(current_dir, "..", "resources", "gui", "img", "copy.svg")
        icon_path_check = path.join(
            current_dir, "..", "resources", "gui", "img", "check.svg"
        )
        self.copy_button.setIcon(QIcon(icon_path))
        self.restore_timer = QTimer(self)

        def restore_text():
            self.copy_button.setIcon(QIcon(icon_path))

        self.restore_timer.timeout.connect(restore_text)

        def copy_text() -> None:
            text = unescape_string(self.label.text())
            QApplication.clipboard().setText(text)
            self.copy_button.setIcon(QIcon(icon_path_check))

            self.restore_timer.start(1000)

        self.copy_button.clicked.connect(copy_text)

        self.addWidget(self.label)
        self.addWidget(self.copy_button, alignment=Qt.AlignLeft)
