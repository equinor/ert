import math
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
from qtpy.QtGui import QPainter, QColorConstants, QPen, QColor
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
        self._delegateWidth = 70
        self._delegateHeight = 70

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

        self._real_view.currentChanged = lambda current, _: self.currentChanged.emit(
            current
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
        colors = index.data(RealJobColorHint)

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        border_pen = QPen()
        border_pen.setColor(QColorConstants.Black)
        border_pen.setWidth(1)

        if option.state & QStyle.State_Selected:

            # selection outline
            select_color = QColorConstants.Blue
            painter.setBrush(select_color)
            painter.setPen(border_pen)
            painter.drawRect(option.rect)

            # job status
            margin = 5
            rect = QRect(
                option.rect.x() + margin,
                option.rect.y() + margin,
                option.rect.width() - (margin * 2),
                option.rect.height() - (margin * 2),
            )
            painter.fillRect(rect, index.data(RealStatusColorHint))

            self._paint_inner_grid(painter, option.rect, colors)

            text_pen = QPen()
            text_pen.setColor(select_color)
            painter.setPen(text_pen)
            painter.drawText(option.rect, Qt.AlignCenter, text)

        else:
            # # job status
            painter.setBrush(index.data(RealStatusColorHint))
            painter.setPen(border_pen)
            painter.drawRect(option.rect)

            self._paint_inner_grid(painter, option.rect, colors)

            text_pen = QPen()
            text_pen.setColor(QColorConstants.Black)
            painter.setPen(text_pen)
            painter.drawText(option.rect, Qt.AlignCenter, text)

        painter.restore()

    def _paint_inner_grid(self, painter: QPainter, rect: QRect, colors) -> None:
        margin = 10
        inner_grid_w = self._size.width() - (margin * 2)
        inner_grid_h = self._size.height() - (margin * 2)
        inner_grid_x = rect.x() + margin
        inner_grid_y = rect.y() + margin
        job_nr = len(colors)
        grid_dim = math.ceil(math.sqrt(job_nr))
        w = math.ceil(inner_grid_w / grid_dim)
        h = math.ceil(inner_grid_h / grid_dim)
        k = 0
        for y in range(grid_dim):
            for x in range(grid_dim):
                x_pos = inner_grid_x + (x * w)
                y_pos = inner_grid_y + (y * h)
                rect = QRect(x_pos, y_pos, w, h)
                if k >= job_nr:
                    color = QColorConstants.Gray
                else:
                    color = colors[k]
                painter.fillRect(rect, color)
                k += 1

    def sizeHint(self, option, index) -> QSize:
        return self._size
