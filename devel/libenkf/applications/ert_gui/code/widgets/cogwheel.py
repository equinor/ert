from PyQt4 import QtGui, QtCore
from widgets.util import frange
import math

class Cogwheel(QtGui.QWidget):

    def __init__(self, color=QtGui.QColor(128, 128, 128), size=64, parent = None):
        QtGui.QWidget.__init__(self, parent)

        self.size = size
        qsize = QtCore.QSize(size, size)
        self.setMaximumSize(qsize)
        self.setMinimumSize(qsize)

        self.color = color
        self.inc = 0
        self.step = 2.5

        self.createCogwheel(size)

        timer = QtCore.QTimer(self)
        self.connect(timer, QtCore.SIGNAL("timeout()"), self, QtCore.SLOT("update()"))
        timer.start(16)

        self.running = False


    def paintEvent(self, paintevent):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        rect = self.contentsRect()

        painter.setPen(QtGui.QColor(255, 255, 255, 0))

        painter.setClipRect(rect)
        painter.translate(rect.center())
        painter.rotate(self.step * self.inc)
        self.drawCogwheel(painter)

        r = (self.size / 2.0) * 0.3
        painter.setBrush(QtGui.QBrush(self.color.light(150)))
        painter.drawEllipse(QtCore.QPointF(0, 0), r, r)

        if self.running:
            self.inc += 1


    def drawCogwheel(self, painter):
        painter.save()
        painter.setBrush(QtGui.QBrush(self.color))
        painter.drawPolygon(QtGui.QPolygonF(self.points), len(self.points))
        painter.restore()


    def createCogwheel(self, size):
        self.points = []
        r1 = (size / 2.0) - 1.0
        r2 = 0.80
        teeth = 9
        out = False
        for t in frange(0.0, 2 * math.pi, 2 * math.pi / (teeth * 2.0)):
            x = r1 * math.cos(t)
            y = r1 * math.sin(t)
            if out:
                self.points.append(QtCore.QPointF(x, y))
                self.points.append(QtCore.QPointF(r2 * x, r2 * y))
            else:
                self.points.append(QtCore.QPointF(r2 * x, r2 * y))
                self.points.append(QtCore.QPointF(x, y))
            out = not out

    def setRunning(self, bool):
        self.running = bool