from typing import Optional, Tuple

from qtpy.QtCore import QRect, QSize, Signal, Slot
from qtpy.QtGui import QColor, QMouseEvent, QPainter, QPaintEvent
from qtpy.QtWidgets import QColorDialog, QFrame


class ColorBox(QFrame):
    colorChanged = Signal(QColor)
    mouseRelease = Signal()

    """A widget that shows a colored box"""

    def __init__(self, size: int = 15) -> None:
        QFrame.__init__(self)
        self.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.setMaximumSize(QSize(size, size))
        self.setMinimumSize(QSize(size, size))

        self._tile_colors = [QColor(255, 255, 255), QColor(200, 200, 255)]
        self._color: QColor = QColor(255, 255, 255)

        self.mouseRelease.connect(self.show_color_dialog)
        self.colorChanged.connect(self.update_color)

    @Slot(QColor)
    def update_color(self, color: QColor) -> None:
        self._color = color
        self.update()

    @Slot()
    def show_color_dialog(self) -> None:
        color_dialog = QColorDialog(self._color, self)
        color_dialog.setWindowTitle("Select color")
        color_dialog.setOption(QColorDialog.ShowAlphaChannel)
        color_dialog.accepted.connect(
            lambda: self.colorChanged.emit(color_dialog.selectedColor())
        )
        color_dialog.open()

    @property
    def color(self) -> QColor:
        return self._color

    @color.setter
    def color(self, color: Tuple[str, float]) -> None:
        new_color = QColor(color[0])
        new_color.setAlphaF(color[1])
        self._color = new_color
        self.update()

    def paintEvent(self, event: Optional[QPaintEvent]) -> None:
        """Paints the box"""
        painter = QPainter(self)
        rect = self.contentsRect()
        tile_count = 3
        tile_size = int(rect.width() / tile_count)
        painter.save()
        painter.translate(rect.x(), rect.y())

        for y in range(tile_count):
            for x in range(tile_count):
                color_index = (y * tile_count + x) % 2
                tile_rect = QRect(x * tile_size, y * tile_size, tile_size, tile_size)
                painter.fillRect(tile_rect, self._tile_colors[color_index])

        painter.restore()
        painter.fillRect(rect, self._color)

        QFrame.paintEvent(self, event)

    def mouseReleaseEvent(self, event: Optional[QMouseEvent]) -> None:
        if event:
            self.mouseRelease.emit()
        return super().mouseReleaseEvent(event)
