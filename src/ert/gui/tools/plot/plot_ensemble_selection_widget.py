from collections.abc import Iterator
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
    QIcon,
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

from .plot_api import EnsembleObject


class EnsembleSelectionWidget(QWidget):
    ensembleSelectionChanged = Signal()

    def __init__(self, ensembles: list[EnsembleObject]) -> None:
        QWidget.__init__(self)
        self.__dndlist = EnsembleSelectListWidget(ensembles[::-1])

        self.__ensemble_layout = QVBoxLayout()
        self.__ensemble_layout.setSpacing(0)
        self.__ensemble_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.__ensemble_layout.addWidget(self.__dndlist)
        self.setLayout(self.__ensemble_layout)
        self.__dndlist.ensembleSelectionListChanged.connect(
            self.ensembleSelectionChanged.emit
        )

    def get_selected_ensembles(self) -> list[EnsembleObject]:
        return self.__dndlist.get_checked_ensembles()


class EnsembleSelectListWidget(QListWidget):
    ensembleSelectionListChanged = Signal()
    MAXIMUM_SELECTED = 5
    MINIMUM_SELECTED = 1

    def __init__(self, ensembles: list[EnsembleObject]) -> None:
        super().__init__()
        self._ensemble_count = 0
        self.setObjectName("ensemble_selector")

        for i, ensemble in enumerate(ensembles):
            it = QListWidgetItem(f"{ensemble.experiment_name} : {ensemble.name}")
            it.setData(Qt.ItemDataRole.UserRole, ensemble)
            it.setData(Qt.ItemDataRole.CheckStateRole, i == 0)
            self.addItem(it)
            self._ensemble_count += 1
            it.setToolTip(
                f"{ensemble.experiment_name} : {ensemble.name}\n"
                "Toggle up to 5 plots or reorder by drag & drop\n"
                "Order determines draw order and color"
            )

        if (viewport := self.viewport()) is not None:
            viewport.setMouseTracking(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.setItemDelegate(CustomItemDelegate())
        self.itemClicked.connect(self.slot_toggle_plot)

    def get_checked_ensembles(self) -> list[EnsembleObject]:
        def _iter() -> Iterator[EnsembleObject]:
            for index in range(self._ensemble_count):
                item = self.item(index)
                assert item is not None
                if item.data(Qt.ItemDataRole.CheckStateRole):
                    yield item.data(Qt.ItemDataRole.UserRole)

        return list(_iter())

    def mouseMoveEvent(self, e: QMouseEvent | None) -> None:
        super().mouseMoveEvent(e)
        if e is not None and self.itemAt(e.pos()):
            self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        else:
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))

    def dropEvent(self, event: QDropEvent | None) -> None:
        super().dropEvent(event)
        self.ensembleSelectionListChanged.emit()

    def slot_toggle_plot(self, item: QListWidgetItem) -> None:
        count = len(self.get_checked_ensembles())
        selected = item.data(Qt.ItemDataRole.CheckStateRole)

        if selected and count > self.MINIMUM_SELECTED:
            item.setData(Qt.ItemDataRole.CheckStateRole, False)
        elif not selected and count < self.MAXIMUM_SELECTED:
            item.setData(Qt.ItemDataRole.CheckStateRole, True)

        self.ensembleSelectionListChanged.emit()


class CustomItemDelegate(QStyledItemDelegate):
    def __init__(self) -> None:
        super().__init__()
        self.swap_pixmap = QIcon("img:reorder.svg").pixmap(QSize(20, 20))

    def sizeHint(self, option: Any, index: Any) -> QSize:
        return QSize(-1, 30)

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
