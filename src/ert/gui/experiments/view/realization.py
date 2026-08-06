from typing import cast, override

from PyQt6.QtCore import (
    QAbstractItemModel,
    QEvent,
    QItemSelectionModel,
    QModelIndex,
    QObject,
    QPoint,
    QSize,
    Qt,
)
from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtGui import QColor, QHelpEvent, QPainter, QPalette, QPen
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QListView,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from ert.gui.experiments.view.iteration_selector import IterationSelector
from ert.gui.model.real_list import RealListModel
from ert.gui.model.snapshot import (
    CallbackStatusMessageRole,
    FMStepColorHint,
    MemoryUsageRole,
    RealIens,
)
from ert.shared.status.utils import byte_with_unit


class RealizationWidget(QWidget):
    triggeredTooltipTextDisplay = Signal(str)
    realizationSelected = Signal(QModelIndex)

    def __init__(self, it: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._iter = it
        self._delegate_size = QSize(90, 90)

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

        self._iteration_selector = IterationSelector(self)
        self._iteration_selector.currentIndexChanged.connect(
            self._on_iteration_selection_changed
        )
        self._selected_realization_by_iteration: dict[int, int] = {}

        selector_layout = QHBoxLayout()
        selector_layout.addWidget(self._iteration_selector)
        selector_layout.addStretch()

        layout = QVBoxLayout()
        layout.addLayout(selector_layout)
        layout.addWidget(self._real_view)

        self.setLayout(layout)

    def _on_current_changed(self, item: QModelIndex, _previous: QModelIndex) -> None:
        if item.isValid():
            self._set_selected_realization(item)

    def _set_selected_realization(self, item: QModelIndex) -> None:
        self._selected_realization_by_iteration[self._iter] = item.row()
        self.realizationSelected.emit(item)

    def set_snapshot_model(self, model: QAbstractItemModel) -> None:
        self._real_list_model = RealListModel(self, self._iter)
        self._real_list_model.setSourceModel(model)

        self._real_view.setModel(self._real_list_model)
        self._real_list_model.setIter(self._iter)

        first_real = self._real_list_model.index(0, 0)
        selection_model = self._real_view.selectionModel()
        if selection_model is None:
            return

        selection_model.currentChanged.connect(self._on_current_changed)
        if first_real.isValid():
            selection_model.select(first_real, QItemSelectionModel.SelectionFlag.Select)

    def hide_iteration_selector(self) -> None:
        self._iteration_selector.hide()

    def add_iteration(self, row: int, label: str) -> None:
        self._iteration_selector.add_iteration(label, row)

    def set_iteration_label(self, row: int, label: str) -> None:
        index = self._iteration_selector.findData(row)
        if index >= 0:
            self._iteration_selector.setItemText(index, label)

    def _on_iteration_selection_changed(self, index: int) -> None:
        if index < 0:
            return

        self._iter = self._iteration_selector.itemData(index)
        self._real_list_model.setIter(self._iter)

        realization_row = self._selected_realization_by_iteration.get(self._iter, 0)
        realization_index = self._real_list_model.index(realization_row, 0)
        selection_model = self._real_view.selectionModel()

        if not realization_index.isValid() or selection_model is None:
            return

        if selection_model.currentIndex() == realization_index:
            self._set_selected_realization(realization_index)
            return

        selection_model.setCurrentIndex(
            realization_index,
            QItemSelectionModel.SelectionFlag.ClearAndSelect,
        )

    def refresh_selected_realization(self) -> None:
        selected_reals = self._real_view.selectedIndexes()
        if selected_reals:
            self._set_selected_realization(selected_reals[0])


class RealizationDelegate(QStyledItemDelegate):
    def __init__(self, size: QSize, parent: QObject) -> None:
        super().__init__(parent)
        self._size = size
        parent.installEventFilter(self)
        self.adjustment_point_for_job_rect_margin = QPoint(-20, -20)
        self._color_black = QColor(0, 0, 0, 180)
        self._color_progress = QColor(50, 173, 230, 200)
        self._color_lightgray = QColor("LightGray").lighter(120)
        self._pen_black = QPen(self._color_black, 2, Qt.PenStyle.SolidLine)

    @override
    def paint(
        self, painter: QPainter | None, option: QStyleOptionViewItem, index: QModelIndex
    ) -> None:
        if painter is None:
            return
        text = index.data(RealIens)
        selected_color, finished_count, total_count = tuple(index.data(FMStepColorHint))

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

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
            factor: int = (
                125
                if selected_color.lighter(125).getRgb() != (255, 255, 255, 255)
                else 110
            )
            selected_color = selected_color.lighter(factor)

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

    @override
    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:
        return self._size

    @override
    def eventFilter(self, object: QObject | None, event: QEvent | None) -> bool:
        if isinstance(event, QHelpEvent) and event.type() == QEvent.Type.ToolTip:
            mouse_pos = event.pos() + self.adjustment_point_for_job_rect_margin
            parent: RealizationWidget = cast(RealizationWidget, self.parent())
            view = parent._real_view
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
                    QToolTip.showText(view.mapToGlobal(mouse_pos), tooltip_text)
                    return True

        return super().eventFilter(object, event)
