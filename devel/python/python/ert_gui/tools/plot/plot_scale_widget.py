from PyQt4.QtCore import pyqtSignal
from PyQt4.QtGui import QWidget, QVBoxLayout, QHBoxLayout, QCheckBox, QLabel, QDoubleSpinBox


class PlotScaleWidget(QWidget):

    Y_MIN = "Y Minimum"
    Y_MAX = "Y Maximum"

    plotScalesChanged = pyqtSignal()

    def __init__(self):
        QWidget.__init__(self)

        self.__scalers = {}

        self.__layout = QVBoxLayout()

        self.addScaler(PlotScaleWidget.Y_MIN)
        self.addScaler(PlotScaleWidget.Y_MAX)
        self.__layout.addStretch()

        self.setLayout(self.__layout)


    def addScaler(self, name):
        widget = QWidget()

        layout = QHBoxLayout()
        layout.setMargin(0)
        widget.setLayout(layout)

        checkbox = QCheckBox()
        checkbox.setChecked(False)
        checkbox.stateChanged.connect(self.updateScalers)
        layout.addWidget(checkbox)

        label = QLabel(name)
        label.setEnabled(False)
        layout.addWidget(label)

        layout.addStretch()

        spinner = QDoubleSpinBox()
        spinner.setEnabled(False)
        spinner.setMinimumWidth(75)
        max = 999999999999
        spinner.setRange(-max,max)
        spinner.valueChanged.connect(self.plotScalesChanged)
        layout.addWidget(spinner)


        self.__layout.addWidget(widget)

        self.__scalers[name] = {"checkbox": checkbox, "label": label, "spinner": spinner}



    def updateScalers(self):
        for scaler in self.__scalers.values():
            checked = scaler["checkbox"].isChecked()
            scaler["label"].setEnabled(checked)
            scaler["spinner"].setEnabled(checked)

        self.plotScalesChanged.emit()


    def getYMin(self):
        scaler = self.__scalers[PlotScaleWidget.Y_MIN]

        if scaler["checkbox"].isChecked():
            return scaler["spinner"].value()
        else:
            return None

    def getYMax(self):
        scaler = self.__scalers[PlotScaleWidget.Y_MAX]

        if scaler["checkbox"].isChecked():
            return scaler["spinner"].value()
        else:
            return None
