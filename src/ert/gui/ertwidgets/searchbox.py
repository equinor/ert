from typing import Any, Optional

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor, QFocusEvent, QKeyEvent
from qtpy.QtWidgets import QLineEdit


class SearchBox(QLineEdit):
    passive_color = QColor(194, 194, 194)

    filterChanged = Signal(object)

    def __init__(self) -> None:
        QLineEdit.__init__(self)

        self.setToolTip("Type to search!")
        self.active_color = self.palette().color(self.foregroundRole())
        self.disable_search = True
        self.presentSearch()
        self.textChanged.connect(self.__emitFilterChanged)

    def __emitFilterChanged(self, _filter: Any) -> None:
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

    def focusInEvent(self, focus_event: Optional[QFocusEvent]) -> None:
        QLineEdit.focusInEvent(self, focus_event)
        self.enterSearch()

    def focusOutEvent(self, focus_event: Optional[QFocusEvent]) -> None:
        QLineEdit.focusOutEvent(self, focus_event)
        self.exitSearch()

    def keyPressEvent(self, key_event: Optional[QKeyEvent]) -> None:
        if key_event and key_event.key() == Qt.Key.Key_Escape:
            self.clear()
            self.clearFocus()
        else:
            QLineEdit.keyPressEvent(self, key_event)
