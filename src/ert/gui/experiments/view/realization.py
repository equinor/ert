from typing import cast, override

from PyQt6.QtCore import (
    QAbstractItemModel,
    QEvent,
    QItemSelectionModel,
    QModelIndex,
    QObject,
    QRect,
    QSize,
    Qt,
)
from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtGui import QColor, QHelpEvent, QPainter, QPalette, QPen
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QListView,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from ert.ensemble_evaluator.state import (
    REALIZATION_STATE_RUNNING,
    REALIZATION_STATE_WAITING,
)
from ert.gui.model.real_list import RealListModel
from ert.gui.model.snapshot import (
    CallbackStatusMessageRole,
    FMStepColorHint,
    MemoryUsageRole,
    RealIens,
    StatusRole,
)
from ert.shared.status.utils import byte_with_unit


class RealizationWidget(QWidget):
    triggeredTooltipTextDisplay = Signal(str)

    def __init__(self, it: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._iter = it
        self._delegate_size = QSize(70, 70)

        self._real_view = QListView(self)
        self._real_view.setViewMode(QListView.ViewMode.IconMode)
        self._real_view.setGridSize(self._delegate_size)
        real_delegate = RealizationDelegate(self._delegate_size, self)
        self._real_view.setMouseTracking(True)
        self._real_view.setItemDelegate(real_delegate)
        self._real_view.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self._real_view.setFlow(QListView.Flow.LeftToRight)
        self._real_view.setWrapping(True)
        self._real_view.setResizeMode(QListView.ResizeMode.Adjust)
        self._real_view.setUniformItemSizes(True)
        self._real_view.setStyleSheet(
            f"QListView {{ background-color: "
            f"{self.palette().color(QPalette.ColorRole.Window).name()}; }}"
        )

        self._real_view.clicked.connect(self._item_clicked)

        layout = QVBoxLayout()
        layout.addWidget(self._real_view)

        self.setLayout(layout)

    # Signal when the user selects another real
    itemClicked = Signal(QModelIndex)

    def _item_clicked(self, item: QModelIndex) -> None:
        self.itemClicked.emit(item)

    def setSnapshotModel(self, model: QAbstractItemModel) -> None:
        self._real_list_model = RealListModel(self, self._iter)
        self._real_list_model.setSourceModel(model)

        self._real_view.setModel(self._real_list_model)
        self._real_list_model.setIter(self._iter)

        first_real = self._real_list_model.index(0, 0)
        selection_model = self._real_view.selectionModel()
        if first_real.isValid() and selection_model:
            selection_model.select(first_real, QItemSelectionModel.SelectionFlag.Select)

    def clearSelection(self) -> None:
        self._real_view.clearSelection()

    def refresh_current_selection(self) -> None:
        selected_reals = self._real_view.selectedIndexes()
        if selected_reals:
            self._item_clicked(selected_reals[0])


class RealizationDelegate(QStyledItemDelegate):
    _STATUS_DOT_DIAMETER = 32
    _PROGRESS_RING_MARGIN = 4
    _PROGRESS_RING_WIDTH = 5

    def __init__(self, item_size: QSize, parent: QObject) -> None:
        super().__init__(parent)
        self._item_size = item_size
        parent.installEventFilter(self)
        self._progress_track_color = QColor(0, 0, 0, 35)
        self._status_dot_outline_pen = QPen(
            QColor(0, 0, 0, 45), 1, Qt.PenStyle.SolidLine
        )

    @override
    def paint(
        self, painter: QPainter | None, option: QStyleOptionViewItem, index: QModelIndex
    ) -> None:
        if painter is None:
            return
        realization_label = index.data(RealIens)
        realization_status = index.data(StatusRole)
        realization_status_color, completed_step_count, total_step_count = tuple(
            index.data(FMStepColorHint)
        )

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        if option.state & QStyle.StateFlag.State_Selected:
            selection_color = QColor(option.palette.color(QPalette.ColorRole.Highlight))
            selection_color.setAlpha(60)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(selection_color)
            painter.drawRoundedRect(option.rect.adjusted(2, 2, -2, -2), 6, 6)

        status_dot_rect = QRect(
            option.rect.center().x() - self._STATUS_DOT_DIAMETER // 2,
            option.rect.top() + 8,
            self._STATUS_DOT_DIAMETER,
            self._STATUS_DOT_DIAMETER,
        )

        if realization_status == REALIZATION_STATE_RUNNING and total_step_count > 0:
            progress_ring_rect = status_dot_rect.adjusted(
                -self._PROGRESS_RING_MARGIN,
                -self._PROGRESS_RING_MARGIN,
                self._PROGRESS_RING_MARGIN,
                self._PROGRESS_RING_MARGIN,
            )
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(self._progress_track_color, self._PROGRESS_RING_WIDTH))
            painter.drawEllipse(progress_ring_rect)
            progress_arc_pen = QPen(realization_status_color, self._PROGRESS_RING_WIDTH)
            progress_arc_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(progress_arc_pen)
            painter.drawArc(
                progress_ring_rect,
                90 * 16,
                -int(360 * 16 * completed_step_count / total_step_count),
            )

        painter.setPen(self._status_dot_outline_pen)
        if realization_status == REALIZATION_STATE_WAITING:
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(realization_status_color, 2))
            painter.drawEllipse(status_dot_rect.adjusted(1, 1, -1, -1))
        else:
            painter.setBrush(realization_status_color)
            painter.drawEllipse(status_dot_rect)

        painter.setPen(option.palette.color(QPalette.ColorRole.Text))
        painter.drawText(
            option.rect.adjusted(0, self._STATUS_DOT_DIAMETER + 14, 0, 0),
            Qt.AlignmentFlag.AlignHCenter,
            realization_label,
        )

        painter.restore()

    @override
    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:
        return self._item_size

    @override
    def eventFilter(self, object: QObject | None, event: QEvent | None) -> bool:
        if isinstance(event, QHelpEvent) and event.type() == QEvent.Type.ToolTip:
            parent: RealizationWidget = cast(RealizationWidget, self.parent())
            view = parent._real_view
            mouse_pos = view.viewport().mapFrom(parent, event.pos())
            index = view.indexAt(mouse_pos)
            if index.isValid():
                tooltip_text = ""
                maximum_memory_usage = index.data(MemoryUsageRole)
                if maximum_memory_usage:
                    tooltip_text += (
                        f"Maximum memory usage:\t{byte_with_unit(maximum_memory_usage)}"
                    )

                if callback_error_msg := index.data(CallbackStatusMessageRole):
                    tooltip_text += callback_error_msg
                if tooltip_text:
                    parent.triggeredTooltipTextDisplay.emit(tooltip_text)
                    QToolTip.showText(
                        view.viewport().mapToGlobal(mouse_pos), tooltip_text
                    )
                    return True

        return super().eventFilter(object, event)
