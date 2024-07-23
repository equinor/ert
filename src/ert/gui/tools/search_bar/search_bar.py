from qtpy import QtCore
from qtpy.QtGui import QBrush, QColor, QTextCharFormat, QTextCursor
from qtpy.QtWidgets import (
    QBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
)


class SearchBar(QLineEdit):
    def __init__(self, text_box: QPlainTextEdit, label: str = "Find"):
        super().__init__()
        self._text_box = text_box
        self._label = QLabel(label, self)
        self.textChanged.connect(self.search_bar_changed)
        self._cursor = self._text_box.textCursor()

    @QtCore.Slot(str)
    def search_bar_changed(self, value: str) -> None:
        self.clear_selection()
        if not value:
            return
        text = self._text_box.toPlainText()
        text_format = QTextCharFormat()
        text_format.setBackground(QBrush(QColor("yellow")))

        if value in text:
            first_instance_pos = text.find(value)
            self.select_text(first_instance_pos, len(value))
            self._cursor.setPosition(0)
            while not self._cursor.atEnd():
                position = self._cursor.position()
                if (
                    position < len(text)
                    and text[position] == value[0]
                    and text[position : position + len(value)] == value
                ):
                    # Check if the entire term matches
                    self._cursor.movePosition(
                        QTextCursor.Right, QTextCursor.KeepAnchor, len(value)
                    )
                    self._cursor.mergeCharFormat(text_format)
                self._cursor.movePosition(
                    QTextCursor.NextCharacter, QTextCursor.MoveAnchor
                )

    def get_layout(self) -> QBoxLayout:
        _layout = QHBoxLayout()
        _layout.addWidget(self._label)
        _layout.addWidget(self)
        return _layout

    def select_text(self, start: int, length: int) -> None:
        self._cursor.setPosition(start)
        self._cursor.movePosition(QTextCursor.Right, QTextCursor.KeepAnchor, length)
        self._text_box.setTextCursor(self._cursor)

    def clear_selection(self) -> None:
        text_format = QTextCharFormat()
        self._cursor.setPosition(QTextCursor.Start)
        self._cursor.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
        text_format.setBackground(QBrush(QColor("white")))
        self._cursor.mergeCharFormat(text_format)
        self._cursor.clearSelection()
        self._text_box.setTextCursor(self._cursor)
