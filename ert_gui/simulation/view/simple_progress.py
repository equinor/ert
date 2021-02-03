from qtpy.QtCore import QRect, QSize, QModelIndex, Qt
from qtpy.QtWidgets import QTreeView, QStyledItemDelegate, QStyleOptionViewItem
from qtpy.QtGui import QPainter, QColor, QFont
from ert_gui.model.snapshot import SimpleProgressRole


class SimpleProgressView(QTreeView):
    def __init__(self, parent=None) -> None:
        super(SimpleProgressView, self).__init__(parent)

        self.setHeaderHidden(True)
        self.setItemsExpandable(False)
        self.setItemDelegate(SimpleProgressDelegate(self))
        self.setRootIsDecorated(False)
        self.setMinimumHeight(15)
        self.setMaximumHeight(15)


class SimpleProgressDelegate(QStyledItemDelegate):
    def __init__(self, parent=None) -> None:
        super(SimpleProgressDelegate, self).__init__(parent)

        self.background_color = QColor(255, 255, 255)
        self.color = QColor(255, 200, 128)

    def paint(self, painter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        painter.save()

        progress = index.data(SimpleProgressRole)
        w = option.rect.width() * progress

        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        painter.fillRect(option.rect, self.background_color)
        painter.fillRect(
            option.rect.x(), option.rect.y(), w, option.rect.height(), self.color
        )

        painter.restore()

    def sizeHint(self, option, index) -> QSize:
        return QSize(15, 15)
