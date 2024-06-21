import math
from typing import Any, Final, Optional, Tuple

from qtpy.QtCore import (
    QAbstractItemModel,
    QEvent,
    QModelIndex,
    QObject,
    QPoint,
    QRect,
    QSize,
    Qt,
    Signal,
)
from qtpy.QtGui import QColor, QColorConstants, QImage, QPainter, QPen
from qtpy.QtWidgets import (
    QAbstractItemView,
    QListView,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from ert.gui.model.real_list import RealListModel
from ert.gui.model.snapshot import MemoryUsageRole, RealJobColorHint, RealLabelHint
from ert.shared.status.utils import byte_with_unit


class RealizationWidget(QWidget):
    def __init__(self, _it: int, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._iter = _it
        self._delegate_size = QSize(90, 90)

        self._real_view = QListView(self)
        self._real_view.setViewMode(QListView.IconMode)
        self._real_view.setGridSize(self._delegate_size)
        real_delegate = RealizationDelegate(self._delegate_size, self)
        self._real_view.setMouseTracking(True)
        self._real_view.setItemDelegate(real_delegate)
        self._real_view.setSelectionMode(QAbstractItemView.SingleSelection)
        self._real_view.setFlow(QListView.LeftToRight)
        self._real_view.setWrapping(True)
        self._real_view.setResizeMode(QListView.Adjust)
        self._real_view.setUniformItemSizes(True)

        def _emit_change(current: QModelIndex, previous: Any) -> None:
            self.currentChanged.emit(current)

        self._real_view.currentChanged = _emit_change  # type: ignore

        layout = QVBoxLayout()
        layout.addWidget(self._real_view)

        self.setLayout(layout)

    # Signal when the user selects another real
    currentChanged = Signal(QModelIndex)

    def setSnapshotModel(self, model: QAbstractItemModel) -> None:
        self._real_list_model = RealListModel(self, self._iter)
        self._real_list_model.setSourceModel(model)

        self._real_view.setModel(self._real_list_model)
        self._real_list_model.setIter(self._iter)

    def clearSelection(self) -> None:
        self._real_view.clearSelection()


class RealizationDelegate(QStyledItemDelegate):
    def __init__(self, width: int, height: int, parent: QObject) -> None:
        super().__init__(parent)
        self._size = size
        self.parent().installEventFilter(self)
        self.adjustment_point_for_job_rect_margin = QPoint(-20, -20)

    def paint(
        self,
        painter: Optional[QPainter],
        option: QStyleOptionViewItem,
        index: QModelIndex,
    ) -> None:
        if painter is None:
            return
        text = index.data(RealLabelHint)
        selected_color, finished_count, total_count = tuple(
            index.data(RealJobColorHint)
        )

        progress_color = QColor(50, 173, 230)

        painter.save()
        painter.setRenderHint(QPainter.TextAntialiasing, True)
        painter.setRenderHint(QPainter.Antialiasing, True)

        if option.state & QStyle.State_Selected:
            selected_color = selected_color.lighter(125)

        percentage_done = int(
            (finished_count * 100.0) / (total_count if total_count > 0 else 1)
        )

        adjusted_rect = option.rect.adjusted(2, 2, -2, -2)
        progressed_step = -int(percentage_done * 57.6)

        painter.setBrush(QColor(QColorConstants.LightGray).lighter(120))
        painter.drawEllipse(adjusted_rect)

        if 0 < percentage_done < 100:
            painter.setBrush(progress_color)
            painter.drawPie(adjusted_rect, 1440, progressed_step)
        else:
            if percentage_done == 100:
                painter.setBrush(progress_color)
            painter.drawEllipse(adjusted_rect)

        painter.setBrush(selected_color)

        adjusted_rect = option.rect.adjusted(7, 7, -7, -7)
        painter.drawEllipse(adjusted_rect)
        pen = QPen(QColorConstants.Black)
        painter.setPen(pen)
        painter.drawText(adjusted_rect, Qt.AlignCenter, text)

        painter.restore()

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:
        return self._size

    def eventFilter(self, object: Optional[QObject], event: Optional[QEvent]) -> bool:
        if event.type() == QEvent.Type.ToolTip:  # type: ignore
            mouse_pos = event.pos() + self.adjustment_point_for_job_rect_margin  # type: ignore
            parent: RealizationWidget = self.parent()  # type: ignore
            view = parent._real_view
            index = view.indexAt(mouse_pos)
            if index.isValid():
                (current_memory_usage, maximum_memory_usage) = index.data(
                    MemoryUsageRole
                )
                if current_memory_usage and maximum_memory_usage:
                    txt = (
                        f"Current memory usage:\t{byte_with_unit(current_memory_usage)}\n"
                        f"Maximum memory usage:\t{byte_with_unit(maximum_memory_usage)}"
                    )
                    QToolTip.showText(view.mapToGlobal(mouse_pos), txt)
                    return True

        return super().eventFilter(object, event)
