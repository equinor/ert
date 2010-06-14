from PyQt4.QtGui import QFrame, QPainter, QColor
from PyQt4.QtCore import QRectF, SIGNAL
from PyQt4.Qt import QApplication, Qt

class ZoomSlider(QFrame):
    def __init__(self, parent=None):
        QFrame.__init__(self, parent)

        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)
        self.setMidLineWidth(3)
        self.setMinimumHeight(21)
        self.setMouseTracking(True)

        self.size = 12

        self.min_value = 0.0
        self.max_value = 1.0

        self.setDefaultColors()
        self.button = Qt.NoButton
        self.selected_marker = 'none'


    def paintEvent(self, paint_event):
        QFrame.paintEvent(self, paint_event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()

        self.min_marker = QRectF(w * self.min_value, 4, self.size, self.size)
        self.max_marker = QRectF(w * self.max_value - self.size - 1, 4, self.size, self.size)

        pen = painter.pen()
        pen.setWidth(0)
        pen.setColor(QApplication.palette().background().color().dark())
        painter.setPen(pen)

        painter.setBrush(self.min_marker_brush)
        painter.drawPie(self.min_marker, 90 * 16, 180 * 16)

        painter.setBrush(self.max_marker_brush)
        painter.drawPie(self.max_marker, 90 * 16, -180 * 16)

    def resizeEvent (self, resize_event):
        QFrame.resizeEvent(self, resize_event)


    def getMinTestMarker(self):
        return QRectF(self.min_marker.left(),
                      self.min_marker.top(),
                      self.min_marker.width() / 2.0,
                      self.min_marker.height())

    def getMaxTestMarker(self):
        return QRectF(self.max_marker.left() + self.max_marker.width() / 2.0,
                      self.max_marker.top(),
                      self.max_marker.width() / 2.0,
                      self.max_marker.height())

    def mouseMoveEvent (self, mouse_event):
        self.setDefaultColors()

        min_test_marker = self.getMinTestMarker()

        if min_test_marker.contains(mouse_event.x(), mouse_event.y()) or self.selected_marker == 'min':
            self.min_marker_brush = self.getDefaultHighlightColor()
        
        if self.selected_marker == 'min':
            value = mouse_event.x() / float(self.width())
            self.setMinValue(value, False)

        max_test_marker = self.getMaxTestMarker()

        if max_test_marker.contains(mouse_event.x(), mouse_event.y()) or self.selected_marker == 'max':
            self.max_marker_brush = self.getDefaultHighlightColor()

        if self.selected_marker == 'max':
            value = mouse_event.x() / float(self.width())
            self.setMaxValue(value, False)

        self.update()


    def mousePressEvent (self, mouse_event):
        if mouse_event.button() == Qt.LeftButton:
            min_test_marker = self.getMinTestMarker()

            if min_test_marker.contains(mouse_event.x(), mouse_event.y()):
                self.selected_marker = 'min'

            max_test_marker = self.getMaxTestMarker()

            if max_test_marker.contains(mouse_event.x(), mouse_event.y()):
                self.selected_marker = 'max'


    def mouseReleaseEvent (self, mouse_event):
        self.selected_marker = 'none'

    def leaveEvent (self, event):
        self.setDefaultColors()

    def getDefaultMarkerColor(self):
        return QApplication.palette().background().color().light(175)

    def getDefaultHighlightColor(self):
        return QApplication.palette().highlight().color()

    def setDefaultColors(self):
        self.min_marker_brush = self.getDefaultMarkerColor()
        self.max_marker_brush = self.getDefaultMarkerColor()
        self.update()

    def setMaxValue(self, max_value, update=True):
        w = float(self.width())
        marker_offset = (self.size + 1) / w

        if not self.max_value == max_value:
            self.max_value = max_value
            if self.max_value - marker_offset <= self.min_value:
                    self.max_value = self.min_value + marker_offset
            if self.max_value > 1.0:
                self.max_value = 1

            self.emit(SIGNAL('zoomValueChanged(float, float)'), self.min_value, self.max_value)

            if update:
                self.update()

    def setMinValue(self, min_value, update=True):
        w = float(self.width())
        marker_offset = (self.size + 1) / w

        if not self.min_value == min_value:
            self.min_value = min_value
            if self.min_value + marker_offset >= self.max_value:
                    self.min_value = self.max_value - marker_offset
            if self.min_value < 0.0:
                self.min_value = 0.0

            self.emit(SIGNAL('zoomValueChanged(float, float)'), self.min_value, self.max_value)

            if update:
                self.update()
