from PyQt6.QtCore import pyqtSlot as Slot
from PyQt6.QtGui import QBrush, QColor, QTextCharFormat, QTextCursor, QTextDocument
from PyQt6.QtWidgets import (
    QBoxLayout,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
)


class SearchBar(QLineEdit):
    def __init__(self, text_box: QPlainTextEdit, label: str = "Find") -> None:
        super().__init__()
        self._text_box = text_box
        self._label = QLabel(label, self)
        self.textChanged.connect(self.search_bar_changed)
        self._cursor = self._text_box.textCursor()

        dialog_buttons = QDialogButtonBox(self)

        self._find_next_button = dialog_buttons.addButton(
            "Find next",
            QDialogButtonBox.ButtonRole.ActionRole,
        )
        assert self._find_next_button is not None
        self._find_next_button.setDefault(True)
        self._find_next_button.clicked.connect(self._find_next)

        self._highlight_all_button = dialog_buttons.addButton(
            "Highlight all",
            QDialogButtonBox.ButtonRole.ActionRole,
        )
        assert self._highlight_all_button is not None
        self._highlight_all_button.clicked.connect(self._highlight_all)

    @Slot(str)
    def search_bar_changed(self, search_term: str) -> None:
        self.clear_selection()
        self._text_box.moveCursor(QTextCursor.MoveOperation.Start)
        self._text_box.find(search_term, QTextDocument.FindFlag.FindCaseSensitively)

    def get_layout(self) -> QBoxLayout:
        layout = QHBoxLayout()
        layout.addWidget(self._label)
        layout.addWidget(self)
        layout.addWidget(self._find_next_button)
        layout.addWidget(self._highlight_all_button)
        return layout

    def select_text(self, start: int, length: int) -> None:
        self._cursor.setPosition(start)
        self._cursor.movePosition(
            QTextCursor.MoveOperation.Right, QTextCursor.MoveMode.KeepAnchor, length
        )
        self._text_box.setTextCursor(self._cursor)

    def clear_selection(self) -> None:
        text_format = QTextCharFormat()
        self._cursor.setPosition(0, QTextCursor.MoveMode.MoveAnchor)
        self._cursor.movePosition(
            QTextCursor.MoveOperation.End, QTextCursor.MoveMode.KeepAnchor
        )
        text_format.setBackground(QBrush(QColor("white")))
        self._cursor.mergeCharFormat(text_format)
        self._cursor.clearSelection()
        self._text_box.setTextCursor(self._cursor)

    def _highlight_all(self) -> None:
        self.clear_selection()
        search_term = self.text()
        if not search_term:
            return

        # The edit block makes the highlight operation faster
        self._cursor.beginEditBlock()

        # Highlight all occurrences
        text = self._text_box.toPlainText()
        highlight_format = QTextCharFormat()
        highlight_format.setBackground(QBrush(QColor("yellow")))

        start = 0
        while (start := text.find(search_term, start)) != -1:
            self._cursor.setPosition(start)
            self._cursor.movePosition(
                QTextCursor.MoveOperation.Right,
                QTextCursor.MoveMode.KeepAnchor,
                len(search_term),
            )
            self._cursor.mergeCharFormat(highlight_format)
            self._cursor.clearSelection()
            start += len(search_term)

        self._cursor.endEditBlock()

    def _find_next(self) -> None:
        search_term = self.text()
        found = self._text_box.find(
            search_term, QTextDocument.FindFlag.FindCaseSensitively
        )
        # If not found, move the cursor to the start and try again
        if not found:
            self._text_box.moveCursor(QTextCursor.MoveOperation.Start)
            self._text_box.find(search_term, QTextDocument.FindFlag.FindCaseSensitively)
