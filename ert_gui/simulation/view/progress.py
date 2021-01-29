from qtpy.QtCore import QRect, QSize, QModelIndex, Qt
from qtpy.QtWidgets import QTreeView, QStyledItemDelegate, QStyleOptionViewItem
from qtpy.QtGui import QPainter, QColor, QFont

from ert_shared.status.entity.state import REAL_STATE_TO_COLOR
from ert_gui.simulation.palette import TOTAL_PROGRESS_COLOR


class ProgressView(QTreeView):
    def __init__(self, parent=None) -> None:
        super(ProgressView, self).__init__(parent)

        self.setHeaderHidden(True)
        self.setItemsExpandable(False)
        self.setItemDelegate(ProgressDelegate(self))
        self.setRootIsDecorated(False)
        self.setMinimumHeight(30)
        self.setMaximumHeight(30)


class ProgressDelegate(QStyledItemDelegate):
    def __init__(self, parent=None) -> None:
        super(ProgressDelegate, self).__init__(parent)

        self.background_color = QColor(200, 210, 210)

    def paint(self, painter, option: QStyleOptionViewItem, index: QModelIndex) -> None:

        nr_reals = index.data()["nr_reals"]
        status = index.data()["status"]
        d = option.rect.width() / nr_reals

        painter.save()

        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        painter.fillRect(option.rect, self.background_color)

        i = 0
        for state, color_ref in REAL_STATE_TO_COLOR.items():

            if state not in status:
                continue

            state_progress = status[state]
            x = option.rect.x() + i * d
            y = option.rect.y()
            w = state_progress * d
            h = option.rect.height()
            color = QColor(*color_ref)

            painter.fillRect(x, y, w, h, color)

            i += state_progress

        painter.restore()

    def sizeHint(self, option, index) -> QSize:
        # x size is ignored
        return QSize(30, 30)
