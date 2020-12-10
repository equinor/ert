from qtpy.QtCore import QRect, QSize, QModelIndex, Qt, QRect, Signal
from qtpy.QtWidgets import (
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QListView,
    QAbstractItemView,
    QStyle,
    QWidget,
    QVBoxLayout,
)
from qtpy.QtGui import QPainter, QColorConstants, QPen
from ert_gui.model.snapshot import (
    RealJobColorHint,
    RealStatusColorHint,
    RealLabelHint,
)
from ert_gui.model.real_list import RealListModel


class RealizationWidget(QWidget):
    def __init__(self, iter: int, parent=None) -> None:
        super(RealizationWidget, self).__init__(parent)

        self._iter = iter
        self._delegateWidth = 50
        self._delegateHeight = 50

        self._real_view = QListView(self)
        self._real_view.setViewMode(QListView.IconMode)
        self._real_view.setGridSize(QSize(self._delegateWidth, self._delegateHeight))
        self._real_view.setItemDelegate(
            RealizationDelegate(self._delegateWidth, self._delegateHeight, self)
        )
        self._real_view.setSelectionMode(QAbstractItemView.SingleSelection)
        self._real_view.setFlow(QListView.LeftToRight)
        self._real_view.setWrapping(True)
        self._real_view.setResizeMode(QListView.Adjust)

        self._real_view.currentChanged = (
            lambda current, previous: self.currentChanged.emit(current)
        )

        layout = QVBoxLayout()
        layout.addWidget(self._real_view)

        self.setLayout(layout)

    # Signal when the user selects another real
    currentChanged = Signal(QModelIndex)

    def setSnapshotModel(self, model) -> None:
        self._real_list_model = RealListModel(self, self._iter)
        self._real_list_model.setSourceModel(model)

        self._real_view.setModel(self._real_list_model)
        self._real_view.model().setIter(self._iter)

    def clearSelection(self) -> None:
        self._real_view.clearSelection()


class RealizationDelegate(QStyledItemDelegate):
    def __init__(self, width, height, parent=None) -> None:
        super(RealizationDelegate, self).__init__(parent)
        self._size = QSize(width, height)

    def paint(self, painter, option: QStyleOptionViewItem, index: QModelIndex) -> None:

        text = index.data(RealLabelHint)

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        border_pen = QPen()
        border_pen.setColor(QColorConstants.Black)
        border_pen.setWidth(1)

        if option.state & QStyle.State_Selected:

            select_color = QColorConstants.Blue

            painter.setBrush(select_color)
            painter.setPen(border_pen)
            painter.drawRect(option.rect)

            margin = 5
            r2 = QRect(
                option.rect.x() + margin,
                option.rect.y() + margin,
                option.rect.width() - (margin * 2),
                option.rect.height() - (margin * 2),
            )
            painter.fillRect(r2, index.data(RealStatusColorHint))

            margin = 10
            r = QRect(
                option.rect.x() + margin,
                option.rect.y() + margin,
                option.rect.width() - (margin * 2),
                option.rect.height() - (margin * 2),
            )
            painter.fillRect(r, index.data(RealJobColorHint))

            text_pen = QPen()
            text_pen.setColor(select_color)
            painter.setPen(text_pen)
            painter.drawText(option.rect, Qt.AlignCenter, text)

        else:
            painter.setBrush(index.data(RealStatusColorHint))
            painter.setPen(border_pen)
            painter.drawRect(option.rect)

            margin = 10
            r = QRect(
                option.rect.x() + margin,
                option.rect.y() + margin,
                option.rect.width() - (margin * 2),
                option.rect.height() - (margin * 2),
            )
            painter.fillRect(r, index.data(RealJobColorHint))

            text_pen = QPen()
            painter.setPen(text_pen)
            painter.drawText(option.rect, Qt.AlignCenter, text)

        painter.restore()

    def sizeHint(self, option, index) -> QSize:
        return self._size
