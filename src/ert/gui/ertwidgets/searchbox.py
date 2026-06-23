from typing import Any, override

from PyQt6.QtCore import QSize, Qt, QTimer
from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtGui import QColor, QCursor, QFocusEvent, QKeyEvent, QMovie, QResizeEvent
from PyQt6.QtWidgets import QApplication, QLabel, QLineEdit, QStyle


class SearchBox(QLineEdit):
    passive_color = QColor(194, 194, 194)

    filterChanged = Signal(object)

    def __init__(self, debounce_timeout: int = 1000) -> None:
        QLineEdit.__init__(self)

        self.setToolTip("Type to search!")
        self.active_color = self.palette().color(self.foregroundRole())
        self.disable_search = True
        self._pending_movie = QMovie("img:loading.gif")
        self._pending_movie.setScaledSize(QSize(16, 16))

        self._pending_indicator = QLabel(self)
        self._pending_indicator.setFixedSize(QSize(16, 16))
        self._pending_indicator.setMovie(self._pending_movie)
        self._pending_indicator.hide()

        self.textChanged.connect(self._start_debounce_timer)

        self._debounce_timeout = debounce_timeout
        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.timeout.connect(self._emit_filter_changed)
        self._search_pending = False

        self.setTextMargins(0, 0, self._pending_indicator.width() + 4, 0)
        self.presentSearch()

    def _show_pending_state(self) -> None:
        self._pending_indicator.show()
        self._pending_movie.start()
        QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))

    def _hide_pending_state(self) -> None:
        self._pending_movie.stop()
        self._pending_indicator.hide()
        QApplication.restoreOverrideCursor()

    def _start_debounce_timer(self, _filter: Any) -> None:
        if self.disable_search:
            return

        if not self._search_pending:
            self._show_pending_state()
            self._search_pending = True

        if not self._debounce_timer.isActive():
            self._debounce_timer.start(self._debounce_timeout)
        self._debounce_timer.setInterval(self._debounce_timeout)

    def _emit_filter_changed(self) -> None:
        self.filterChanged.emit(self.filter())
        if self._search_pending:
            self._hide_pending_state()
            self._search_pending = False

    def filter(self) -> str:
        if self.disable_search:
            return ""
        return str(self.text())

    def presentSearch(self) -> None:
        """Is called to present the greyed out search"""
        self.disable_search = True
        if self._search_pending:
            self._hide_pending_state()
            self._search_pending = False
        self.setText("Search")
        palette = self.palette()
        palette.setColor(self.foregroundRole(), self.passive_color)
        self.setPalette(palette)

    def activateSearch(self) -> None:
        """Is called to remove the greyed out search"""
        self.setText("")
        self.disable_search = False
        palette = self.palette()
        palette.setColor(self.foregroundRole(), self.active_color)
        self.setPalette(palette)

    def enterSearch(self) -> None:
        """Called when the line edit gets the focus"""
        if str(self.text()) == "Search":
            self.activateSearch()

    def exitSearch(self) -> None:
        """Called when the line edit loses focus"""
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
    def resizeEvent(self, a0: QResizeEvent | None) -> None:
        style = self.style()
        if style:
            frame_width = style.pixelMetric(QStyle.PixelMetric.PM_DefaultFrameWidth)
            self._pending_indicator.move(
                self.rect().right() - frame_width - self._pending_indicator.width(),
                int((self.height() - self._pending_indicator.height()) / 2),
            )

        QLineEdit.resizeEvent(self, a0)

    @override
    def keyPressEvent(self, a0: QKeyEvent | None) -> None:
        if a0 and a0.key() == Qt.Key.Key_Escape:
            self.clear()
            self.clearFocus()
        else:
            QLineEdit.keyPressEvent(self, a0)
