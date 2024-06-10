import math

from PyQt5.QtWidgets import QListView
from qtpy.QtCore import QModelIndex, Qt
from qtpy.QtGui import QColor, QPainter, QPalette
from qtpy.QtWidgets import (
    QApplication,
    QFrame,
    QStyledItemDelegate,
    QStyleOptionViewItem,
)

from ert.ensemble_evaluator.state import REAL_STATE_TO_COLOR
from ert.gui.model.progress_proxy import ProgressRole


class LegendView(QListView):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setItemDelegate(LegendDelegate(self))
        self.setFixedHeight(30)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.NoFrame)


class LegendDelegate(QStyledItemDelegate):
    @staticmethod
    def paint(painter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        data = index.data(ProgressRole)
        nr_reals = data["nr_reals"] if data else 0
        status = data["status"] if data else {}

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        background_color = QApplication.palette().color(QPalette.Window)
        painter.fillRect(
            option.rect.x(), option.rect.y(), option.rect.width(), 30, background_color
        )

        total_states = len(REAL_STATE_TO_COLOR.items())
        delta = math.ceil(option.rect.width() / total_states)
        x_offset = 0
        y = option.rect.y() + 5
        h = option.rect.height()
        w = delta - 25

        for state, color_ref in REAL_STATE_TO_COLOR.items():
            x = x_offset
            painter.setBrush(QColor(*color_ref))
            painter.drawRect(x, y, 20, 20)
            state_progress = status.get(state, 0)
            text = f"{state} ({state_progress}/{nr_reals})"
            painter.drawText(x + 25, y, w, h, Qt.AlignLeft, text)
            x_offset += delta

        painter.restore()
