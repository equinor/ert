from PyQt4 import QtGui, QtCore

class LegendMarker(QtGui.QWidget):
    def __init__(self, color, parent = None):
        QtGui.QWidget.__init__(self, parent)

        self.setMaximumSize(QtCore.QSize(12, 12))
        self.setMinimumSize(QtCore.QSize(12, 12))

        self.color = color

    def paintEvent(self, paintevent):
        painter = QtGui.QPainter(self)

        rect = self.contentsRect()
        rect.setWidth(rect.width() - 1)
        rect.setHeight(rect.height() - 1)
        painter.drawRect(rect)

        rect.setX(rect.x() + 1)
        rect.setY(rect.y() + 1)
        painter.fillRect(rect, self.color)

class Legend(QtGui.QHBoxLayout):
    def __init__(self, legend, color, parent=None):
        QtGui.QHBoxLayout.__init__(self, parent)

        legendMarker = LegendMarker(color, parent)
        self.addWidget(legendMarker)
        self.addWidget(QtGui.QLabel(legend))
        legendMarker.setToolTip(legend)