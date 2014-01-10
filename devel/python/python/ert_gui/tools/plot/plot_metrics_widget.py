from PyQt4.QtCore import pyqtSignal, Qt
from PyQt4.QtGui import QWidget, QVBoxLayout, QHBoxLayout, QCheckBox, QLabel, QDoubleSpinBox
from ert_gui.tools.plot import ReportStepWidget


class PlotMetricsWidget(QWidget):

    VALUE_MIN = "Value Minimum"
    VALUE_MAX = "Value Maximum"

    DEPTH_MIN = "Depth Minimum"
    DEPTH_MAX = "Depth Maximum"

    TIME_MIN = "Time Minimum"
    TIME_MAX = "Time Maximum"

    plotScalesChanged = pyqtSignal()
    reportStepTimeChanged = pyqtSignal()

    def __init__(self):
        QWidget.__init__(self)

        self.__scalers = {}

        self.__layout = QVBoxLayout()

        self.addScaler(PlotMetricsWidget.VALUE_MIN)
        self.addScaler(PlotMetricsWidget.VALUE_MAX)
        self.__layout.addSpacing(10)

        self.__report_step_widget = ReportStepWidget()
        self.__report_step_widget.reportStepTimeSelected.connect(self.reportStepTimeChanged)
        self.__layout.addWidget(self.__report_step_widget)

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
        # spinner.valueChanged.connect(self.plotScalesChanged)
        spinner.editingFinished.connect(self.plotScalesChanged)
        layout.addWidget(spinner)


        self.__layout.addWidget(widget)

        self.__scalers[name] = {"checkbox": checkbox, "label": label, "spinner": spinner}



    def updateScalers(self):
        for scaler in self.__scalers.values():
            checked = scaler["checkbox"].isChecked()
            scaler["label"].setEnabled(checked)
            scaler["spinner"].setEnabled(checked)

        self.plotScalesChanged.emit()


    def getValueMin(self):
        scaler = self.__scalers[PlotMetricsWidget.VALUE_MIN]

        if scaler["checkbox"].isChecked():
            return scaler["spinner"].value()
        else:
            return None

    def getValueMax(self):
        scaler = self.__scalers[PlotMetricsWidget.VALUE_MAX]

        if scaler["checkbox"].isChecked():
            return scaler["spinner"].value()
        else:
            return None


    def setValueScales(self, value_min, value_max):
        self.__updateScale(PlotMetricsWidget.VALUE_MIN, value_min)
        self.__updateScale(PlotMetricsWidget.VALUE_MAX, value_max)
        self.plotScalesChanged.emit()


    def __updateScale(self, name, value):
        scaler = self.__scalers[name]
        if value is None:
            scaler["checkbox"].blockSignals(True)
            scaler["checkbox"].setCheckState(Qt.Unchecked)
            scaler["checkbox"].blockSignals(False)
        else:
            scaler["checkbox"].blockSignals(True)
            scaler["checkbox"].setCheckState(Qt.Checked)
            scaler["checkbox"].blockSignals(False)

            scaler["spinner"].blockSignals(True)
            scaler["spinner"].setValue(value)
            scaler["spinner"].blockSignals(False)

        checked = scaler["checkbox"].isChecked()
        scaler["label"].setEnabled(checked)
        scaler["spinner"].setEnabled(checked)



    def getSelectedReportStepTime(self):
        """ @rtype: ctime """
        return self.__report_step_widget.getSelectedValue().ctime()