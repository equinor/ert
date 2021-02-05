import math
from qtpy.QtCore import QSize, QModelIndex, Qt
from qtpy.QtWidgets import QTreeView, QStyledItemDelegate, QStyleOptionViewItem
from qtpy.QtGui import QPainter, QColor
from ert_shared.status.entity.state import REAL_STATE_TO_COLOR
from ert_gui.model.progress_proxy import ProgressRole


class LegendView(QTreeView):
    def __init__(self, parent=None) -> None:
        super(LegendView, self).__init__(parent)

        self.setHeaderHidden(True)
        self.setItemsExpandable(False)
        self.setItemDelegate(LegendDelegate(self))
        self.setRootIsDecorated(False)
        self.setMinimumHeight(30)
        self.setMaximumHeight(30)


class LegendDelegate(QStyledItemDelegate):
    def __init__(self, parent=None) -> None:
        super(LegendDelegate, self).__init__(parent)

    def paint(self, painter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        data = index.data(ProgressRole)
        if data is None:
            return
        nr_reals = data["nr_reals"]
        status = data["status"]

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        total_states = len(REAL_STATE_TO_COLOR.items())
        d = math.ceil(option.rect.width() / total_states)
        x_pos = 0
        for state, color_ref in REAL_STATE_TO_COLOR.items():

            if state not in status:
                state_progress = 0
            else:
                state_progress = status[state]

            x = x_pos
            y = option.rect.y()
            w = d
            h = option.rect.height()
            color = QColor(*color_ref)

            painter.fillRect(x, y, w, h, color)
            painter.drawText(
                x, y, w, h, Qt.AlignCenter, f"{state} ({state_progress}/{nr_reals})"
            )

            x_pos += d

        painter.restore()

    def sizeHint(self, option, index) -> QSize:
        return QSize(30, 30)
