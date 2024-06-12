from __future__ import annotations

import math

from qtpy.QtCore import QModelIndex, QSize, Qt
from qtpy.QtGui import QColor, QPainter
from qtpy.QtWidgets import (
    QProgressBar,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from ert.ensemble_evaluator.state import REAL_STATE_TO_COLOR
from ert.gui.model.snapshot import ProgressRole


class ProgressView(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self._progress_tree_view = QTreeView(self)
        self._progress_tree_view.setHeaderHidden(True)
        self._progress_tree_view.setItemsExpandable(False)
        self._progress_tree_view.setItemDelegate(ProgressDelegate(self))
        self._progress_tree_view.setRootIsDecorated(False)
        self._progress_tree_view.setFixedHeight(30)
        self._progress_tree_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self._progress_bar = QProgressBar(self)
        self._progress_bar.setRange(0, 0)
        self._progress_bar.setFixedHeight(30)
        self._progress_bar.setVisible(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._progress_tree_view)
        layout.addWidget(self._progress_bar)

        self.setLayout(layout)
        self.setFixedHeight(30)

    def setModel(self, model) -> None:
        self._progress_tree_view.setModel(model)

    def set_active_progress(self, enable: bool = True) -> None:
        self._progress_bar.setVisible(not enable)
        self._progress_tree_view.setVisible(enable)


class ProgressDelegate(QStyledItemDelegate):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self.background_color = QColor(200, 210, 210)

    def paint(self, painter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        data = index.data(ProgressRole)
        if data is None:
            return

        nr_reals: int = data["nr_reals"]
        status: dict[str, int] = data["status"]
        delta = option.rect.width() / nr_reals if nr_reals else 1

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.fillRect(option.rect, self.background_color)

        i = 0
        y = option.rect.y()
        h = option.rect.height()

        for state, color_ref in REAL_STATE_TO_COLOR.items():
            if state in status:
                state_progress = status[state]
                x = math.ceil(option.rect.x() + i * delta)
                w = math.ceil(state_progress * delta)
                painter.fillRect(x, y, w, h, QColor(*color_ref))
                i += state_progress

        painter.restore()

    @staticmethod
    def sizeHint(option, index) -> QSize:
        return index.data(role=Qt.SizeHintRole)
