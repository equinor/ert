from collections import deque
from collections.abc import Iterator
from enum import IntEnum
from typing import Any

from PyQt6.QtCore import (
    QModelIndex,
    QSize,
    Qt,
)
from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QCursor,
    QDropEvent,
    QMouseEvent,
    QPainter,
    QPen,
)
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QListWidget,
    QListWidgetItem,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QVBoxLayout,
    QWidget,
)
from typing_extensions import override

from ert.gui.icon_utils import load_icon
from ert.gui.utils import is_everest_application

from .plot_api import EnsembleObject


class EnsembleSelectionWidget(QWidget):
    ensembleSelectionChanged = Signal()

    def __init__(
        self,
        ensembles: list[EnsembleObject],
        number_of_plot_colors: int,
    ) -> None:
        QWidget.__init__(self)
        self._selected_ensembles = EnsembleSelectListWidget(
            ensembles, number_of_plot_colors
        )

        self._ensemble_layout = QVBoxLayout()
        self._ensemble_layout.setSpacing(0)
        self._ensemble_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._ensemble_layout.addWidget(self._selected_ensembles)
        self.setLayout(self._ensemble_layout)
        self._selected_ensembles.ensembleSelectionListChanged.connect(
            self.ensembleSelectionChanged.emit
        )

    def apply_ensemble_filtering(
        self, require_func_eval: bool, require_gradient: bool
    ) -> None:
        self._selected_ensembles.apply_ensemble_filtering(
            require_func_eval, require_gradient
        )

    def get_selected_ensembles(self) -> list[EnsembleObject]:
        return self._selected_ensembles.get_checked_ensembles()

    def get_selected_ensembles_color_indexes(self) -> list[int]:
        return self._selected_ensembles.get_checked_color_indexes()


class EnsembleSelectListWidgetItemDataRole(IntEnum):
    ENSEMBLE = Qt.ItemDataRole.UserRole
    COLOR_INDEX = Qt.ItemDataRole.UserRole + 1


class EnsembleSelectListWidget(QListWidget):
    ensembleSelectionListChanged = Signal()
    MAXIMUM_SELECTED = 5
    MINIMUM_SELECTED = 1

    def __init__(
        self,
        ensembles: list[EnsembleObject],
        number_of_plot_colors: int,
    ) -> None:
        super().__init__()
        self.available_colors = deque(
            range(max(number_of_plot_colors, self.MAXIMUM_SELECTED))
        )
        self._ensemble_count = 0
        self.setObjectName("ensemble_selector")
        sorted_ensembles = sorted(
            ensembles, key=lambda ens: ens.started_at, reverse=True
        )
        is_everest = is_everest_application()
        cutoff = self.MAXIMUM_SELECTED if is_everest else self.MINIMUM_SELECTED
        for i, ensemble in enumerate(sorted_ensembles):
            item_text = (
                f"{ensemble.experiment_name} : {ensemble.name}"
                if not is_everest
                else ensemble.name
            )
            it = QListWidgetItem(item_text)
            it.setData(EnsembleSelectListWidgetItemDataRole.ENSEMBLE, ensemble)
            it.setData(
                EnsembleSelectListWidgetItemDataRole.COLOR_INDEX,
                self.assign_available_color(None) if i < cutoff else None,
            )
            it.setData(Qt.ItemDataRole.CheckStateRole, i < cutoff)
            self.addItem(it)
            self._ensemble_count += 1
            it.setToolTip(
                f"{item_text}\n"
                f"Toggle up to {self.MAXIMUM_SELECTED} plots or reorder by"
                "drag & drop\n"
                f"Order determines draw order and color"
            )

        if (viewport := self.viewport()) is not None:
            viewport.setMouseTracking(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.setItemDelegate(CustomItemDelegate())
        self.itemClicked.connect(self.slot_toggle_plot)

    def apply_ensemble_filtering(
        self, require_func_eval: bool, require_gradient: bool
    ) -> None:
        for index in range(self._ensemble_count):
            if (item := self.item(index)) is None:
                raise ValueError(
                    f"Expected ensemble at index {index} in ensemble selection list"
                )
            ensemble: EnsembleObject = item.data(
                EnsembleSelectListWidgetItemDataRole.ENSEMBLE
            )
            hidden = (require_func_eval and not ensemble.has_func_eval) or (
                require_gradient and not ensemble.has_gradient
            )
            item.setHidden(hidden)
            if hidden and item.data(Qt.ItemDataRole.CheckStateRole):
                self.release_color(
                    item.data(EnsembleSelectListWidgetItemDataRole.COLOR_INDEX)
                )
                item.setData(Qt.ItemDataRole.CheckStateRole, False)

    def get_checked_ensembles(self) -> list[EnsembleObject]:
        def _iter() -> Iterator[EnsembleObject]:
            for index in range(self._ensemble_count):
                if (item := self.item(index)) is None:
                    raise ValueError(
                        f"Expected ensemble at index {index} in ensemble selection list"
                    )
                if item.data(Qt.ItemDataRole.CheckStateRole):
                    yield item.data(EnsembleSelectListWidgetItemDataRole.ENSEMBLE)

        return list(_iter())

    def get_checked_color_indexes(self) -> list[int]:
        def _iter() -> Iterator[int]:
            for index in range(self._ensemble_count):
                if (item := self.item(index)) is None:
                    raise ValueError(
                        f"Expected ensemble at index {index} in ensemble selection list"
                    )
                if item.data(Qt.ItemDataRole.CheckStateRole):
                    yield item.data(EnsembleSelectListWidgetItemDataRole.COLOR_INDEX)

        return list(_iter())

    @override
    def mouseMoveEvent(self, e: QMouseEvent | None) -> None:
        super().mouseMoveEvent(e)
        if e is not None and self.itemAt(e.pos()):
            self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        else:
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))

    @override
    def dropEvent(self, event: QDropEvent | None) -> None:
        super().dropEvent(event)
        self.ensembleSelectionListChanged.emit()

    def slot_toggle_plot(self, item: QListWidgetItem) -> None:
        count = len(self.get_checked_ensembles())
        selected = item.data(Qt.ItemDataRole.CheckStateRole)

        if selected and count > self.MINIMUM_SELECTED:
            self.release_color(
                item.data(EnsembleSelectListWidgetItemDataRole.COLOR_INDEX)
            )
            item.setData(Qt.ItemDataRole.CheckStateRole, False)
        elif not selected and count < self.MAXIMUM_SELECTED:
            item.setData(
                EnsembleSelectListWidgetItemDataRole.COLOR_INDEX,
                self.assign_available_color(
                    item.data(EnsembleSelectListWidgetItemDataRole.COLOR_INDEX)
                ),
            )
            item.setData(Qt.ItemDataRole.CheckStateRole, True)

        self.ensembleSelectionListChanged.emit()

    def assign_available_color(self, current_color_index: int | None) -> int:
        if (
            current_color_index is not None
            and current_color_index in self.available_colors
        ):
            self.available_colors.remove(current_color_index)
            return current_color_index
        return self.available_colors.popleft()

    def release_color(self, current_color_index: int | None) -> None:
        if current_color_index is not None:
            self.available_colors.append(current_color_index)


class CustomItemDelegate(QStyledItemDelegate):
    def __init__(self) -> None:
        super().__init__()
        self.swap_pixmap = load_icon("reorder.svg").pixmap(QSize(20, 20))

    @override
    def sizeHint(self, option: Any, index: Any) -> QSize:
        return QSize(-1, 30)

    @override
    def paint(
        self, painter: QPainter | None, option: QStyleOptionViewItem, index: QModelIndex
    ) -> None:
        if painter is None:
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        pen_color = QColor("black")
        background_color = QColor("lightgray")
        selected_background_color = QColor("lightblue")

        rect = option.rect.adjusted(2, 2, -2, -2)
        painter.setPen(QPen(pen_color))

        if index.data(Qt.ItemDataRole.CheckStateRole):
            painter.setBrush(QBrush(selected_background_color))
        else:
            painter.setBrush(QBrush(background_color))

        painter.drawRect(rect)

        icon_logical_width = int(
            self.swap_pixmap.width() / self.swap_pixmap.devicePixelRatio()
        )
        icon_logical_height = int(
            self.swap_pixmap.height() / self.swap_pixmap.devicePixelRatio()
        )

        text_rect = rect.adjusted(2 * icon_logical_width, 4, -4, -4)
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignLeft, index.data())

        cursor_x = int(option.rect.left() + icon_logical_width / 2)
        cursor_y = int(option.rect.center().y() - (icon_logical_height / 2))
        painter.drawPixmap(cursor_x, cursor_y, self.swap_pixmap)
