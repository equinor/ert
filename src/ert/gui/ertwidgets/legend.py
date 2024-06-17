from typing import Optional

from qtpy.QtCore import QSize
from qtpy.QtGui import QColor, QPainter, QPaintEvent
from qtpy.QtWidgets import QHBoxLayout, QLabel, QWidget


class LegendMarker(QWidget):
    """A widget that shows a colored box"""

    def __init__(self, color: QColor):
        QWidget.__init__(self)

        self.setMaximumSize(QSize(12, 12))
        self.setMinimumSize(QSize(12, 12))

        self.color = color

    def paintEvent(self, a0: Optional[QPaintEvent]) -> None:
        painter = QPainter(self)

        rect = self.contentsRect()
        rect.setWidth(rect.width() - 1)
        rect.setHeight(rect.height() - 1)
        painter.drawRect(rect)

        rect.setX(rect.x() + 1)
        rect.setY(rect.y() + 1)
        painter.fillRect(rect, self.color)


class Legend(QWidget):
    """Combines a LegendMarker with a label"""

    def __init__(self, legend: Optional[str], color: QColor):
        QWidget.__init__(self)

        self.setMinimumWidth(140)
        self.setMaximumHeight(25)

        self.legend = legend

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        self.legend_marker = LegendMarker(color)
        self.legend_marker.setToolTip(legend)

        layout.addWidget(self.legend_marker)
        self.legend_label = QLabel(legend)
        layout.addWidget(self.legend_label)
        layout.addStretch()

        self.setLayout(layout)
