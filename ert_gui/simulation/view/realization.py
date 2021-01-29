from qtpy.QtCore import QRect, QSize, QModelIndex, Qt
from qtpy.QtWidgets import QStyledItemDelegate, QStyleOptionViewItem, QListView, QAbstractItemView, QStyle
from qtpy.QtGui import QPainter, QColor, QFont, QColorConstants, QPen

from ert_shared.status.entity.state import REAL_STATE_TO_COLOR
from ert_gui.model.snapshot import NodeRole



class RealizationView(QListView):
    def __init__(self, parent=None) -> None:
        super(RealizationView, self).__init__(parent)

        self._delegateWidth= 50
        self._delegateHeight= 50
        self.setViewMode(QListView.IconMode)
        self.setGridSize(QSize(self._delegateWidth, self._delegateHeight))
        self.setItemDelegate(RealizationDelegate(self._delegateWidth, self._delegateHeight, self))
        self.clicked.connect(self._select_real)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setFlow(QListView.LeftToRight)
        self.setWrapping(True)
        self.setResizeMode(QListView.Adjust)

    def _select_real(self, e) -> None:
        print(f"select {e}")


class RealizationDelegate(QStyledItemDelegate):
    def __init__(self, width, height, parent = None) -> None:
        super(RealizationDelegate, self).__init__(parent)
        self._size = QSize(width, height)

    def paint(self, painter, option : QStyleOptionViewItem, index : QModelIndex) -> None:
        node = index.data(NodeRole)
        text= f"{node.id}"
        status= node.data["status"]

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        fill_color = QColor(*REAL_STATE_TO_COLOR[status])
        pen_color = QColor(QColor(11, 11, 11))
        if option.state & QStyle.State_Selected:
            fill_color = QColorConstants.Blue
            pen_color = QColorConstants.White

        painter.setPen(pen_color)
        painter.setBrush(fill_color)

        painter.drawRect(option.rect)
        painter.drawText(option.rect, Qt.AlignCenter, text)

        painter.restore()


    def sizeHint(self, option, index) -> QSize:
        return self._size