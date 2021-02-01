from qtpy.QtCore import QRect, QSize, QModelIndex, Qt, QRect
from qtpy.QtWidgets import (
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QListView,
    QAbstractItemView,
    QStyle,
)
from qtpy.QtGui import QPainter, QColor, QFont, QColorConstants, QPen

from ert_gui.model.snapshot import Node, NodeRole, RealJobColorHint, RealStatusColorHint, RealLabelHint


class RealizationView(QListView):
    def __init__(self, parent=None) -> None:
        super(RealizationView, self).__init__(parent)

        self._delegateWidth = 50
        self._delegateHeight = 50
        self.setViewMode(QListView.IconMode)
        self.setGridSize(QSize(self._delegateWidth, self._delegateHeight))
        self.setItemDelegate(
            RealizationDelegate(self._delegateWidth, self._delegateHeight, self)
        )
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setFlow(QListView.LeftToRight)
        self.setWrapping(True)
        self.setResizeMode(QListView.Adjust)



class RealizationDelegate(QStyledItemDelegate):
    def __init__(self, width, height, parent=None) -> None:
        super(RealizationDelegate, self).__init__(parent)
        self._size = QSize(width, height)

    def paint(self, painter, option: QStyleOptionViewItem, index: QModelIndex) -> None:

        text= index.data(RealLabelHint)

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        if option.state & QStyle.State_Selected:
            border_pen = QPen()
            border_pen.setColor(QColorConstants.Black)
            border_pen.setWidth(1)

            painter.setBrush(QColorConstants.Blue)
            painter.setPen(border_pen)
            painter.drawRect(option.rect)

            text_pen = QPen()
            text_pen.setColor(QColorConstants.White)
            painter.setPen(text_pen)
            painter.drawText(option.rect, Qt.AlignCenter, text)

        else:
            # border + job status
            border_pen = QPen()
            border_pen.setColor(QColorConstants.Black)
            border_pen.setWidth(1)
            painter.setBrush(index.data(RealJobColorHint))
            painter.setPen(border_pen)
            painter.drawRect(option.rect)

            # real status
            margin = 10
            r = QRect(
                option.rect.x() + margin,
                option.rect.y() + margin,
                option.rect.width() - (margin * 2),
                option.rect.height() - (margin * 2),
            )
            painter.fillRect(r, index.data(RealStatusColorHint))

            # text
            text_pen = QPen()
            painter.setPen(text_pen)
            painter.drawText(option.rect, Qt.AlignCenter, text)

        painter.restore()

    def sizeHint(self, option, index) -> QSize:
        return self._size
