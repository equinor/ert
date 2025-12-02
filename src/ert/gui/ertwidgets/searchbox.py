from typing import Any

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtGui import QColor, QFocusEvent, QKeyEvent
from PyQt6.QtWidgets import QLineEdit
from typing_extensions import override


class SearchBox(QLineEdit):
    passive_color = QColor(194, 194, 194)

    filterChanged = Signal(object)

    def __init__(self, debounce_timeout: int = 1000) -> None:
        QLineEdit.__init__(self)

        self.setToolTip("Type to search!")
        self.active_color = self.palette().color(self.foregroundRole())
        self.disable_search = True
        self.presentSearch()
        self.textChanged.connect(self._start_debounce_timer)
        self._debounce_timout = debounce_timeout
        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.timeout.connect(self._emit_filter_changed)

    def _start_debounce_timer(self, _filter: Any) -> None:
        if not self._debounce_timer.isActive():
            self._debounce_timer.start(self._debounce_timout)
        self._debounce_timer.setInterval(self._debounce_timout)

    def _emit_filter_changed(self) -> None:
        self.filterChanged.emit(self.filter())

    def filter(self) -> str:
        if self.disable_search:
            return ""
        else:
            return str(self.text())

    def presentSearch(self) -> None:
        """Is called to present the greyed out search"""
        self.disable_search = True
        self.setText("Search")
        palette = self.palette()
        palette.setColor(self.foregroundRole(), self.passive_color)
        self.setPalette(palette)

    def activateSearch(self) -> None:
        """Is called to remove the greyed out search"""
        self.disable_search = False
        self.setText("")
        palette = self.palette()
        palette.setColor(self.foregroundRole(), self.active_color)
        self.setPalette(palette)

    def enterSearch(self) -> None:
        """Called when the line edit gets the focus"""
        if str(self.text()) == "Search":
            self.activateSearch()

    def exitSearch(self) -> None:
        """Called when the line edit looses focus"""
        if not self.text():
            self.presentSearch()

    @override
    def focusInEvent(self, a0: QFocusEvent | None) -> None:
        QLineEdit.focusInEvent(self, a0)
        self.enterSearch()

    @override
    def focusOutEvent(self, a0: QFocusEvent | None) -> None:
        QLineEdit.focusOutEvent(self, a0)
        self.exitSearch()

    @override
    def keyPressEvent(self, a0: QKeyEvent | None) -> None:
        if a0 and a0.key() == Qt.Key.Key_Escape:
            self.clear()
            self.clearFocus()
        else:
            QLineEdit.keyPressEvent(self, a0)
