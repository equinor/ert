import math
from typing import Final

from qtpy.QtCore import QModelIndex, QRect, QSize, Qt, Signal
from qtpy.QtGui import QColor, QColorConstants, QImage, QPainter, QPen
from qtpy.QtWidgets import (
    QAbstractItemView,
    QListView,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QVBoxLayout,
    QWidget,
)

from ert.ensemble_evaluator import state
from ert.gui.model.real_list import RealListModel
from ert.gui.model.snapshot import RealJobColorHint, RealLabelHint, RealStatusColorHint

COLOR_UNKNOWN: Final[QColor] = QColor(*state.COLOR_UNKNOWN)
COLOR_WAITING: Final[QColor] = QColor(*state.COLOR_WAITING)
COLOR_PENDING: Final[QColor] = QColor(*state.COLOR_PENDING)
COLOR_RUNNING: Final[QColor] = QColor(*state.COLOR_RUNNING)
COLOR_FINISHED: Final[QColor] = QColor(*state.COLOR_FINISHED)
COLOR_FAILED: Final[QColor] = QColor(*state.COLOR_FAILED)


class RealizationWidget(QWidget):
    def __init__(self, _it: int, parent=None) -> None:
        super().__init__(parent)

        self._iter = _it
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
        self._real_view.setUniformItemSizes(True)

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


# This singleton is shared among all instances of RealizationDelegate
_image_cache = {}


class RealizationDelegate(QStyledItemDelegate):
    def __init__(self, width, height, parent=None) -> None:
        super().__init__(parent)
        self._size = QSize(width, height)

    def paint(self, painter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        text = index.data(RealLabelHint)
        colors = tuple(index.data(RealJobColorHint))
        queue_status_color = index.data(RealStatusColorHint)

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, False)
        painter.setRenderHint(QPainter.TextAntialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, False)

        border_pen = QPen()
        border_pen.setColor(QColorConstants.Black)
        border_pen.setWidth(1)

        painter.setBrush(QColorConstants.Blue)
        painter.setPen(border_pen)
        painter.drawRect(option.rect)

        margin = 0
        if option.state & QStyle.State_Selected:
            margin = 5

        realization_status_rect = QRect(
            option.rect.x() + margin,
            option.rect.y() + margin,
            option.rect.width() - (margin * 2),
            option.rect.height() - (margin * 2),
        )

        realization_color_status = determine_realization_status_color(colors)

        if (
            queue_status_color == COLOR_WAITING
            and realization_color_status == COLOR_PENDING
        ):  # corner case where queue system provide more information than jobs
            realization_color_status = COLOR_WAITING
            colors = [x if x != COLOR_PENDING else COLOR_WAITING for x in colors]

        painter.setBrush(realization_color_status)
        painter.drawRect(realization_status_rect)

        job_rect_margin = 10
        job_rect = QRect(
            option.rect.x() + job_rect_margin,
            option.rect.y() + job_rect_margin,
            option.rect.width() - (job_rect_margin * 2),
            option.rect.height() - (job_rect_margin * 2),
        )

        self._paint_inner_grid(painter, job_rect, colors)

        text_pen = QPen()
        text_pen.setColor(QColorConstants.Black)
        painter.setPen(text_pen)
        painter.drawText(option.rect, Qt.AlignCenter, text)

        painter.restore()

    def _paint_inner_grid(self, painter: QPainter, rect: QRect, colors) -> None:
        job_nr = len(colors)
        grid_dim = math.ceil(math.sqrt(job_nr))
        k = 0

        colors_hash = hash(tuple(color.name() for color in colors))
        if colors_hash not in _image_cache:
            foreground_image = QImage(grid_dim, grid_dim, QImage.Format_ARGB32)
            foreground_image.fill(QColorConstants.Gray)

            for y in range(grid_dim):
                for x in range(grid_dim):
                    color = QColorConstants.Gray if k >= job_nr else colors[k]
                    foreground_image.setPixel(x, y, color.rgb())
                    k += 1
            _image_cache[colors_hash] = foreground_image
        else:
            foreground_image = _image_cache[colors_hash]

        painter.drawImage(rect, foreground_image)

    def sizeHint(self, option, index) -> QSize:
        return self._size


def determine_realization_status_color(job_colors) -> QColor:
    states = [
        COLOR_UNKNOWN,
        COLOR_FINISHED,
        COLOR_WAITING,
        COLOR_PENDING,
    ]

    current_index = 0

    for color in job_colors:
        if color in [COLOR_RUNNING, COLOR_FAILED]:
            return color
        elif states.index(color) > current_index:
            current_index = states.index(color)

    return states[current_index]
