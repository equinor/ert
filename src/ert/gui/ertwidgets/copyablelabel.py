from os import path
from threading import Timer

from qtpy.QtCore import Qt
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QApplication, QHBoxLayout, QLabel, QPushButton, QSizePolicy

from ert.gui.ertwidgets import addHelpToWidget

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

    def __init__(self, text, executeAddHelpToWidget=True) -> None:
        super().__init__()

        self.label = QLabel(f"<b>{escape_string(text)}</b>")
        self.label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.copy_button = QPushButton("")
        self.copy_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.copy_button.setStyleSheet(
            "QPushButton { padding: 8px; max-width: 150px; min-width: 30px; }"
        )
        icon_path = path.join(current_dir, "..", "resources", "gui", "img", "copy.svg")
        self.copy_button.setIcon(QIcon(icon_path))

        def copy_text() -> None:
            text = unescape_string(self.label.text())
            QApplication.clipboard().setText(text)
            self.copy_button.setText("copied!")

            def restore_text():
                self.copy_button.setText("")

            Timer(1.0, restore_text).start()

        self.copy_button.clicked.connect(copy_text)

        self.addWidget(self.label)
        self.addWidget(self.copy_button)

        if executeAddHelpToWidget:
            addHelpToWidget(self.label, "")
