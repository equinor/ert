from PyQt4.QtCore import pyqtSignal
from PyQt4.QtGui import QWidget, QVBoxLayout
from ert_gui.tools.plot import ReportStepWidget, ScaleTracker, PlotScalesWidget

class PlotMetricsWidget(QWidget):

    VALUE_MIN = "Value Minimum"
    VALUE_MAX = "Value Maximum"

    DEPTH_MIN = "Depth Minimum"
    DEPTH_MAX = "Depth Maximum"

    TIME_MIN = "Time Minimum"
    TIME_MAX = "Time Maximum"

    plotSettingsChanged = pyqtSignal()

    def __init__(self):
        QWidget.__init__(self)
        self.__layout = QVBoxLayout()

        self.__data_type_key = None
        self.__trackers = {}
        self.__spinners = {}

        self.__layout.addWidget(self.addScaler("value_min", self.VALUE_MIN))
        self.__layout.addWidget(self.addScaler("value_max", self.VALUE_MAX))
        self.__layout.addWidget(self.addScaler("time_min", self.TIME_MIN, True,True))
        self.__layout.addWidget(self.addScaler("time_max", self.TIME_MAX, True))
        self.__layout.addWidget(self.addScaler("depth_min", self.DEPTH_MIN))
        self.__layout.addWidget(self.addScaler("depth_max", self.DEPTH_MAX))

        self.__layout.addSpacing(10)

        self.__report_step_widget = ReportStepWidget()
        self.__report_step_widget.reportStepTimeSelected.connect(self.plotSettingsChanged)
        self.__layout.addWidget(self.__report_step_widget)

        self.__layout.addStretch()

        self.setLayout(self.__layout)

    def updateTrackers(self, values):
        type_key = values["type_key"]
        enabled = values["enabled"]
        value = values["value"]
        tracker = self.__trackers[type_key]
        tracker.setValues(self.__data_type_key, value, enabled)
        self.plotSettingsChanged.emit()

    def addScaler(self, type_key, title, time_spinner=False,min_value=False):
        valueSpinner = PlotScalesWidget(type_key, title, time_spinner,min_value)
        valueSpinner.plotScaleChanged.connect(self.updateTrackers)
        self.__spinners[type_key] = valueSpinner
        self.__trackers[type_key] = ScaleTracker(type_key)
        return valueSpinner

    def getDataKeyType(self):
        return self.__data_type_key

    def setDataKeyType(self, data_key_type):
        self.__data_type_key = data_key_type
        for key in self.__spinners:
            scaler = self.__spinners[key]
            self.blockSignals(True)
            values = {"type_key": key, "enabled": self.getIsEnabled(key), "value": self.getValue(key)}
            scaler.setValues(values)
            self.blockSignals(False)

    def getValue(self, type_key):
        if type_key in self.__trackers:
            return self.__trackers[type_key].getScaleValue(self.__data_type_key)
        else:
            return None

    def getIsEnabled(self, type_key):
        if type_key in self.__trackers:
            return self.__trackers[type_key].isEnabled(self.__data_type_key)
        else:
            return None

    def getSettings(self):
        settings = {
            "value_min" : self.getValue("value_min"),
            "value_max" : self.getValue("value_max"),
            "time_min" : self.getValue("time_min"),
            "time_max" : self.getValue("time_max"),
            "depth_min" : self.getValue("depth_min"),
            "depth_max" : self.getValue("depth_max"),
            "report_step_time" : self.__report_step_widget.getSelectedValue().ctime()
        }
        return settings