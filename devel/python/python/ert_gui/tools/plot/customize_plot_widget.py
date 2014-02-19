from PyQt4.QtCore import pyqtSignal, Qt
from PyQt4.QtGui import QWidget, QVBoxLayout, QHBoxLayout, QCheckBox, QLabel


class CustomizePlotWidget(QWidget):

    customPlotSettingsChanged = pyqtSignal(dict)

    def __init__(self):
        QWidget.__init__(self)

        self.__custom = {}

        self.__layout = QVBoxLayout()

        self.addCheckBox("error_bar_only", "Show only error bars", False)

        self.__layout.addStretch()

        self.setLayout(self.__layout)


    def emitChange(self):
        self.customPlotSettingsChanged.emit(self.__custom)

    def addCheckBox(self, name, description, default_value):
        checkbox = QCheckBox(description)
        checkbox.setChecked(default_value)
        self.__custom[name] = default_value

        def toggle(checked):
            self.__custom[name] = checked
            self.emitChange()

        checkbox.toggled.connect(toggle)

        self.__layout.addWidget(checkbox)



