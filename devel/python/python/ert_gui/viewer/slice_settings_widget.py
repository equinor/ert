from PyQt4.QtCore import pyqtSignal, QString
from PyQt4.QtGui import QWidget, QCheckBox, QVBoxLayout, QSpinBox, QFormLayout, QComboBox


class SliceSettingsWidget(QWidget):

    inactiveCellsHidden = pyqtSignal(bool)
    regionToggling = pyqtSignal(bool)
    toggleOrthographicProjection = pyqtSignal(bool)
    toggleLighting = pyqtSignal(bool)
    currentSliceChanged = pyqtSignal(int)
    colorScalesChanged = pyqtSignal(QString)
    toggleInterpolation = pyqtSignal(bool)

    def __init__(self, max_slice_count, color_scales):
        QWidget.__init__(self)

        self.setMinimumWidth(250)

        self.__layout = QFormLayout()

        self.addCheckBox("Hide inactive cells", self.inactiveCellsHidden)
        self.addSpinBox("Slice #", 0, max_slice_count - 1, self.currentSliceChanged)
        self.addCheckBox("Orthographic proj.", self.toggleOrthographicProjection)
        self.addCheckBox("Lighting", self.toggleLighting)
        self.addComboBox("Color scale", self.colorScalesChanged, color_scales)
        self.addCheckBox("Region scaling", self.regionToggling)
        self.addCheckBox("Interpolate data", self.toggleInterpolation)

        self.setLayout(self.__layout)


    def addCheckBox(self, label, signal):
        checkbox = QCheckBox()
        checkbox.toggled.connect(signal)
        self.__layout.addRow(label, checkbox)


    def addSpinBox(self, label, min_value, max_value, signal):
        spinner = QSpinBox()
        spinner.setMinimum(min_value)
        spinner.setMaximum(max_value)
        spinner.valueChanged[int].connect(signal)
        self.__layout.addRow(label, spinner)

    def addComboBox(self, label, signal, choices):
        combo = QComboBox()
        combo.addItems(choices)
        combo.currentIndexChanged[QString].connect(signal)
        self.__layout.addRow(label, combo)