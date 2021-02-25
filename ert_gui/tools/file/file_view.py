from bisect import bisect_left, bisect_right

from builtins import range

from qtpy.QtCore import QModelIndex, QPoint, QRect, QSize, Slot
from qtpy.QtGui import QFont, QFontDatabase, QPainter, QPalette, QRegion
from qtpy.QtWidgets import QAbstractItemView, QStyle, QStyleOptionViewItem


class FileView(QAbstractItemView):
    def __init__(self, parent=None):
        super(FileView, self).__init__(parent)
        self.setSelectionMode(self.NoSelection)
        self._line_offsets = [0]
        self._line_sizes = []
        self._size = QSize()
        self._force_follow = False

        self._init_font()

    def paintEvent(self, event):
        painter = QPainter(self.viewport())

        view_rect = self._translate(event.rect())
        if view_rect.y() < 0:
            rect = QRect()
            rect.setHeight(self.viewport().height() + view_rect.y())
            rect.setWidth(self.viewport().width())

            brush = self.palette().brush(QPalette.Disabled, QPalette.Window)
            painter.fillRect(rect, brush)

        for index in self._intersecting_rows(view_rect):
            option = self._style_option(index)
            delegate = self.itemDelegate(index)
            delegate.paint(painter, option, index)

    def setSelection(self, _rect, _flags):
        """setSelection is a pure virtual member function of QAbstractItemView"""

    def scrollTo(self, _index, _hint):
        """scrollTo is a pure virtual member function of QAbstractItemView"""

    def indexAt(self, point):
        """indexAt is a pure virtual member function of QAbstractItemView"""
        return self.model().index(self._resolve_row(point), 0)

    def moveCursor(self, _action, _modifiers):
        """moveCursor is a pure virtual member function of QAbstractItemView"""
        return QModelIndex()

    def visualRect(self, index):
        """visualRect is a pure virtual member function of QAbstractItemView"""
        if index.row() < 0 or index.row() >= len(self._line_sizes):
            return QRect()
        point = QPoint(
            -self.horizontalOffset(),
            self._line_offsets[index.row()] - self.verticalOffset(),
        )
        return QRect(point, self._line_sizes[index.row()])

    def visualRegionForSelection(self, _selection):
        """visualRegionForSelection is a pure virtual member function of QAbstractItemView"""
        return QRegion()

    def horizontalOffset(self):
        """horizontalOffset is a pure virtual member function of QAbstractItemView"""
        return self.horizontalScrollBar().value()

    def verticalOffset(self):
        """verticalOffset is a pure virtual member function of QAbstractItemView"""
        if self._force_follow:
            return self._size.height() - self.viewport().height()
        else:
            return self.verticalScrollBar().value()

    def isIndexHidden(self, _index):
        """isIndexHidden is a pure virtual member function of QAbstractItemView"""
        return False

    @Slot(QModelIndex, int, int)
    def rowsInserted(self, parent, first, last):
        """rowsInserted is an overriden slot of QAbstractItemView"""
        if first != len(self._line_sizes):
            raise NotImplementedError("FileView can only be appended to")
        model = self.model()

        follow = (
            self._force_follow
            or self.verticalOffset() == self.verticalScrollBar().maximum()
        )

        for idx in range(first, last):
            end = self._size.height()
            index = model.index(idx, 0)

            option = self._style_option(index)
            delegate = self.itemDelegate(index)
            size = delegate.sizeHint(option, index)

            self._line_sizes.append(size)
            self._line_offsets.append(end + size.height())
            self._size.setWidth(max(size.width(), self._size.width()))
            self._size.setHeight(end + size.height())
        self.updateGeometries()

        if follow:
            self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())
        self.viewport().update()

    @Slot()
    def updateGeometries(self):
        """updateGeometries is an overridden slot of QAbstractItemView"""
        horizontal_max = self._size.width() - self.viewport().width()
        self.horizontalScrollBar().setRange(0, horizontal_max)

        if self._force_follow:
            self.verticalScrollBar().setRange(0, 0)
        else:
            vertical_max = self._size.height() - self.viewport().height()
            self.verticalScrollBar().setRange(0, vertical_max)

        QAbstractItemView.updateGeometries(self)

    @Slot(bool)
    def enable_follow_mode(self, follow=True):
        """Enables or disabled follow mode, as in tail -f"""
        self._force_follow = follow
        self.updateGeometries()
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())
        self.viewport().update()

    def _intersecting_rows(self, rect):
        """Get rows that intersect with rect"""
        model = self.model()
        first = max(0, bisect_left(self._line_offsets, rect.top()) - 1)
        last = min(
            len(self._line_sizes), bisect_right(self._line_offsets, rect.bottom())
        )

        return [model.index(row, 0) for row in range(first, last)]

    def _resolve_row(self, point):
        """Get row that is at point"""
        y = point.y() + self.verticalOffset()
        return max(0, bisect_left(self._line_offsets, y) - 1)

    def _init_font(self):
        # There isn't a standard way of getting the system default monospace
        # font in Qt4 (it was introduced in Qt5.2). If QFontDatabase.FixedFont
        # exists, then we can assume that this functionality exists and ask for
        # the correct font directly. Otherwise we ask for a font that doesn't
        # exist and specify our requirements. Qt then finds an existing font
        # that best matches our parameters.
        if hasattr(QFontDatabase, "systemFont") and hasattr(QFontDatabase, "FixedFont"):
            font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        else:
            font = QFont("")
            font.setFixedPitch(True)
            font.setStyleHint(QFont.Monospace)
        self.setFont(font)

    def _style_option(self, index):
        """Get default style option for index"""
        option = QStyleOptionViewItem()
        option.font = self.font()
        option.rect = self.visualRect(index)
        option.state = QStyle.State_Enabled
        return option

    def _translate(self, rect):
        """Translate rect from viewport to document coordinates"""
        return rect.translated(self.horizontalOffset(), self.verticalOffset())
