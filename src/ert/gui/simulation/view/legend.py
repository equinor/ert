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
    def paint(self, painter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        data = index.data(ProgressRole)
        nr_reals = 0
        status = {}
        if data:
            nr_reals = data["nr_reals"]
            status = data["status"]

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        background_color = QApplication.palette().color(QPalette.Window)
        painter.fillRect(
            option.rect.x(), option.rect.y(), option.rect.width(), 30, background_color
        )

        total_states = len(REAL_STATE_TO_COLOR.items())
        d = math.ceil(option.rect.width() / total_states)
        x_pos = 0
        for state, color_ref in REAL_STATE_TO_COLOR.items():
            state_progress = 0
            if state in status:
                state_progress = status[state]

            x = x_pos
            y = option.rect.y()
            w = d
            h = option.rect.height()
            margin = 5

            painter.setBrush(QColor(*color_ref))
            painter.drawRect(x, y + margin, 20, 20)
            painter.drawText(
                x + 25,
                y + margin,
                w - 25,
                h,
                Qt.AlignLeft,
                f"{state} ({state_progress}/{nr_reals})",
            )
            x_pos += d

        painter.restore()
