from typing import List

from PyQt5.QtCore import QSize
from PyQt5.QtGui import QBrush, QColor, QCursor, QIcon, QPainter, QPen
from PyQt5.QtWidgets import QAbstractItemView, QStyledItemDelegate
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)


class EnsembleSelectionWidget(QWidget):
    ensembleSelectionChanged = Signal()

    def __init__(self, ensemble_names: List[str]):
        QWidget.__init__(self)
        self.__dndlist = EnsembleSelectListWidget(ensemble_names)

        self.__ensemble_layout = QVBoxLayout()
        self.__ensemble_layout.setSpacing(0)
        self.__ensemble_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.__ensemble_layout.addWidget(self.__dndlist)
        self.setLayout(self.__ensemble_layout)
        self.__dndlist.ensembleSelectionListChanged.connect(
            self.ensembleSelectionChanged.emit
        )

    def getPlotEnsembleNames(self) -> List[str]:
        return self.__dndlist.get_checked_ensemble_plot_names()


class EnsembleSelectListWidget(QListWidget):
    ensembleSelectionListChanged = Signal()
    MAXIMUM_SELECTED = 5
    MINIMUM_SELECTED = 1

    def __init__(self, ensembles):
        super().__init__()
        self._ensembles = ensembles
        self.setObjectName("ensemble_selector")

        for i, ensemble in enumerate(self._ensembles):
            it = QListWidgetItem(ensemble)
            it.setData(Qt.ItemDataRole.UserRole, i == 0)
            self.addItem(it)

        self.viewport().setMouseTracking(True)
        self.setDragDropMode(QAbstractItemView.InternalMove)
        self.setItemDelegate(CustomItemDelegate())
        self.itemClicked.connect(self.slot_toggle_plot)
        self.setToolTip(
            "Select/deselect [1,5] or reorder plots\nOrder determines draw order and color"
        )

    def get_checked_ensemble_plot_names(self) -> List[str]:
        return [
            self.item(index).text()
            for index in range(self.count())
            if self.item(index).data(Qt.ItemDataRole.UserRole)
        ]

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        if self.itemAt(event.pos()):
            self.setCursor(QCursor(Qt.PointingHandCursor))
        else:
            self.setCursor(QCursor(Qt.ArrowCursor))

    def dropEvent(self, e):
        super().dropEvent(e)
        self.ensembleSelectionListChanged.emit()

    def slot_toggle_plot(self, item: QListWidgetItem):
        count = len(self.get_checked_ensemble_plot_names())
        selected = item.data(Qt.ItemDataRole.UserRole)

        if selected and count > self.MINIMUM_SELECTED:
            item.setData(Qt.ItemDataRole.UserRole, False)
        elif not selected and count < self.MAXIMUM_SELECTED:
            item.setData(Qt.ItemDataRole.UserRole, True)

        self.ensembleSelectionListChanged.emit()


class CustomItemDelegate(QStyledItemDelegate):
    def __init__(self):
        super().__init__()
        self.swap_pixmap = QIcon("img:swap_vertical.svg").pixmap(QSize(20, 20))

    def sizeHint(self, option, index):
        return QSize(-1, 30)

    def paint(self, painter, option, index):
        painter.setRenderHint(QPainter.Antialiasing)

        pen_color = QColor("black")
        background_color = QColor("lightgray")
        selected_background_color = QColor("lightblue")

        rect = option.rect.adjusted(2, 2, -2, -2)
        painter.setPen(QPen(pen_color))

        if index.data(Qt.ItemDataRole.UserRole):
            painter.setBrush(QBrush(selected_background_color))
        else:
            painter.setBrush(QBrush(background_color))

        painter.drawRect(rect)

        text_rect = rect.adjusted(4, 4, -4, -4)
        painter.drawText(text_rect, Qt.AlignHCenter, index.data())

        cursor_x = option.rect.right() - self.swap_pixmap.width() - 5
        cursor_y = int(option.rect.center().y() - (self.swap_pixmap.height() / 2))
        painter.drawPixmap(cursor_x, cursor_y, self.swap_pixmap)
