from typing import Optional, Tuple

from qtpy.QtCore import QRect, QSize, Signal
from qtpy.QtGui import QColor, QMouseEvent, QPainter, QPaintEvent
from qtpy.QtWidgets import QColorDialog, QFrame


class ColorBox(QFrame):
    colorChanged = Signal(QColor)

    """A widget that shows a colored box"""

    def __init__(self, color: QColor, size: int = 15) -> None:
        QFrame.__init__(self)
        self.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.setMaximumSize(QSize(size, size))
        self.setMinimumSize(QSize(size, size))

        self._tile_colors = [QColor(255, 255, 255), QColor(200, 200, 255)]
        self._color = color

    def paintEvent(self, a0: Optional[QPaintEvent]) -> None:
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

        QFrame.paintEvent(self, a0)

    def mouseReleaseEvent(self, a0: Optional[QMouseEvent]) -> None:
        color = QColorDialog.getColor(
            self._color, self, "Select color", QColorDialog.ShowAlphaChannel
        )

        if color.isValid():
            self._color = color
            self.update()
            self.colorChanged.emit(self._color)

    @property
    def color(self) -> QColor:
        return self._color

    @color.setter
    def color(self, color: Tuple[str, float]) -> None:
        new_color = QColor(color[0])
        new_color.setAlphaF(color[1])
        self._color = new_color
        self.update()
