from typing import Any, Optional

from qtpy.QtCore import (
    QAbstractItemModel,
    QEvent,
    QModelIndex,
    QObject,
    QPoint,
    QSize,
    Qt,
    Signal,
)
from qtpy.QtGui import QColor, QColorConstants, QPainter, QPalette, QPen
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
        self._real_view.setStyleSheet(
            f"QListView {{ background-color: {self.palette().color(QPalette.Window).name()}; }}"
        )

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
    def __init__(self, size: QSize, parent: QObject) -> None:
        super().__init__(parent)
        self._size = size
        parent.installEventFilter(self)
        self.adjustment_point_for_job_rect_margin = QPoint(-20, -20)
        self._color_black = QColor(0, 0, 0, 180)
        self._color_progress = QColor(50, 173, 230, 200)
        self._color_lightgray = QColor(QColorConstants.LightGray).lighter(120)
        self._pen_black = QPen(self._color_black, 2, Qt.PenStyle.SolidLine)

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

        painter.save()
        painter.setRenderHint(QPainter.TextAntialiasing, True)
        painter.setRenderHint(QPainter.Antialiasing, True)

        percentage_done = (
            100 if total_count < 1 else int((finished_count * 100.0) / total_count)
        )

        painter.setPen(self._pen_black)
        adjusted_rect = option.rect.adjusted(2, 2, -2, -2)

        painter.setBrush(
            self._color_progress if percentage_done == 100 else self._color_lightgray
        )
        painter.drawEllipse(adjusted_rect)

        if 0 < percentage_done < 100:
            painter.setBrush(self._color_progress)
            painter.drawPie(adjusted_rect, 1440, -int(percentage_done * 57.6))

        if option.state & QStyle.StateFlag.State_Selected:
            selected_color = selected_color.lighter(125)

        painter.setBrush(selected_color)
        adjusted_rect = option.rect.adjusted(7, 7, -7, -7)
        painter.drawEllipse(adjusted_rect)

        font = painter.font()
        font.setBold(True)
        painter.setFont(font)

        adj_rect = option.rect.adjusted(0, 20, 0, 0)
        painter.drawText(adj_rect, Qt.AlignmentFlag.AlignHCenter, text)
        adj_rect = option.rect.adjusted(0, 45, 0, 0)
        painter.drawText(
            adj_rect, Qt.AlignmentFlag.AlignHCenter, f"{finished_count} / {total_count}"
        )

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
