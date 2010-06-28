from PyQt4.QtCore import QObject, QSize
from PyQt4.Qt import SIGNAL
from PyQt4.QtGui import QFormLayout, QFrame, QComboBox, QHBoxLayout, QDoubleSpinBox, QWidget, QPainter, QColor, QColorDialog
from PyQt4.QtGui import QCheckBox

class PlotConfig(object):

    def __init__(self, name, linestyle="-", marker="", color=(0.0, 0.0, 0.0), alpha=0.75, zorder=1, picker=None, visible=True):
        self._name = name
        self._linestyle = linestyle
        self._marker = marker
        self._color = color
        self._alpha = alpha
        self._is_visible = visible
        self._z_order = zorder
        self._picker = picker

        self.signal_handler = QObject()

    def notify(self):
        self.signal_handler.emit(SIGNAL('plotConfigChanged(PlotConfig)'), self)

    def get_name(self):
        return self._name

    name = property(get_name)

    def hasStyle(self):
        return not self.style == ""

    def get_style(self):
        return (str(self._marker) + str(self._linestyle)).strip()

    style = property(get_style)

    def setLinestyle(self, linestyle):
        self._linestyle = linestyle
        self.notify()

    def getLinestyle(self):
        return self._linestyle

    linestyle = property(getLinestyle, setLinestyle)

    def setMarker(self, marker):
        self._marker = marker
        self.notify()

    def getMarker(self):
        return self._marker

    marker = property(getMarker, setMarker)

    def setAlpha(self, alpha):
        self._alpha = alpha
        self.notify()

    def getAlpha(self):
        return self._alpha

    alpha = property(getAlpha, setAlpha)

    def setColor(self, color):
        self._color = color
        self.notify()

    def getColor(self):
        return self._color

    color = property(getColor, setColor)


    def set_is_visible(self, is_visible):
        self._is_visible = is_visible
        self.notify()

    def is_visible(self):
        return self._is_visible

    is_visible = property(is_visible, set_is_visible)

    def set_z_order(self, z_order):
        self._z_order = z_order
        self.notify()

    def get_z_order(self):
        return self._z_order

    z_order = property(get_z_order, set_z_order)

    def setPicker(self, picker):
        self._picker = picker
        self.notify()

    def getPicker(self):
        return self._picker

    picker = property(getPicker, setPicker)



class PlotConfigPanel(QFrame):
    plot_marker_styles = ["", ".", ",", "o", "*", "s", "+", "x", "p", "h", "H", "D", "d"]
    plot_line_styles = ["", "-", "--", "-.", ":"]

    def __init__(self, plot_config):
        QFrame.__init__(self)
        self.plot_config = plot_config
        self.connect(plot_config.signal_handler, SIGNAL('plotConfigChanged(PlotConfig)'), self.fetchValues)

        layout = QFormLayout()
        layout.setRowWrapPolicy(QFormLayout.WrapLongRows)

        self.chk_visible = QCheckBox()
        layout.addRow("Visible:", self.chk_visible)
        self.connect(self.chk_visible, SIGNAL('stateChanged(int)'), self.setVisibleState)

        self.plot_linestyle = QComboBox()
        self.plot_linestyle.addItems(self.plot_line_styles)
        self.connect(self.plot_linestyle, SIGNAL("currentIndexChanged(QString)"), self.setLineStyle)
        layout.addRow("Line style:", self.plot_linestyle)

        self.plot_marker_style = QComboBox()
        self.plot_marker_style.addItems(self.plot_marker_styles)
        self.connect(self.plot_marker_style, SIGNAL("currentIndexChanged(QString)"), self.setMarker)
        layout.addRow("Marker style:", self.plot_marker_style)



        self.alpha_spinner = QDoubleSpinBox(self)
        self.alpha_spinner.setMinimum(0.0)
        self.alpha_spinner.setMaximum(1.0)
        self.alpha_spinner.setDecimals(3)
        self.alpha_spinner.setSingleStep(0.01)

        self.connect(self.alpha_spinner, SIGNAL('valueChanged(double)'), self.setAlpha)
        layout.addRow("Blend factor:", self.alpha_spinner)

        self.color_picker = ColorPicker(plot_config)
        layout.addRow("Color:", self.color_picker)

        self.setLayout(layout)
        self.fetchValues(plot_config)

    def fetchValues(self, plot_config):
        self.plot_config = plot_config

        #block signals to avoid updating the incoming plot_config 

        state = self.plot_linestyle.blockSignals(True)
        linestyle_index = self.plot_line_styles.index(self.plot_config.linestyle)
        self.plot_linestyle.setCurrentIndex(linestyle_index)
        self.plot_linestyle.blockSignals(state)

        state = self.plot_marker_style.blockSignals(True)
        marker_index = self.plot_marker_styles.index(self.plot_config.marker)
        self.plot_marker_style.setCurrentIndex(marker_index)
        self.plot_marker_style.blockSignals(state)

        state = self.alpha_spinner.blockSignals(True)
        self.alpha_spinner.setValue(self.plot_config.alpha)
        self.alpha_spinner.blockSignals(state)

        state = self.chk_visible.blockSignals(True)
        self.chk_visible.setChecked(self.plot_config.is_visible)
        self.chk_visible.blockSignals(state)

        self.color_picker.update()

    def setLineStyle(self, linestyle):
        self.plot_config.linestyle = linestyle

    def setMarker(self, marker):
        self.plot_config.marker = marker

    def setAlpha(self, alpha):
        self.plot_config.alpha = alpha

    def setVisibleState(self, state):
        self.plot_config.is_visible = state == 2
   


class ColorPicker(QWidget):
    """A widget that shows a colored box"""
    color_dialog = QColorDialog()

    def __init__(self, plot_config):
        QWidget.__init__(self)

        self.plot_config = plot_config

        size = 20
        self.setMaximumSize(QSize(size, size))
        self.setMinimumSize(QSize(size, size))
        self.setToolTip("Click to change color!")

    def paintEvent(self, paintevent):
        """Paints the box"""
        painter = QPainter(self)

        rect = self.contentsRect()
        rect.setWidth(rect.width() - 1)
        rect.setHeight(rect.height() - 1)
        painter.drawRect(rect)

        rect.setX(rect.x() + 1)
        rect.setY(rect.y() + 1)
        painter.fillRect(rect, self.getColor())

    def setColor(self, color):
        self.plot_config.color = (color.redF(), color.greenF(), color.blueF())
        self.update()

    def getColor(self):
        color = self.plot_config.color
        return QColor(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))

    def mouseDoubleClickEvent(self, event):
        self.color_dialog.setCurrentColor(self.getColor())
        self.color_dialog.exec_()
        self.setColor(self.color_dialog.selectedColor())

    def mousePressEvent(self, event):
        self.color_dialog.setCurrentColor(self.getColor())
        self.color_dialog.exec_()
        self.setColor(self.color_dialog.selectedColor())

