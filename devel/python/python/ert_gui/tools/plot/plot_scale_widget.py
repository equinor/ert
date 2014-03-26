from PyQt4.QtCore import pyqtSignal
from PyQt4.QtGui import QWidget, QHBoxLayout, QCheckBox, QDoubleSpinBox
from ert_gui.models.connectors.plot import ReportStepsModel
from ert_gui.widgets.list_spin_box import ListSpinBox

class PlotScalesWidget(QWidget):

    plotScaleChanged = pyqtSignal(dict)

    def __init__(self, type_key, title, time_spinner, min_value, minimum=None, maximum=None):
        QWidget.__init__(self)

        if minimum is None:
            minimum = -999999999999

        if maximum is None:
            maximum = 999999999999


        self.__time_spinner = time_spinner
        if time_spinner:
            self.__time_map = ReportStepsModel().getList()
            self.__time_index_map = {}
            for index in range(len(self.__time_map)):
                time = self.__time_map[index]
                self.__time_index_map[time] = index

        self.__type_key = type_key
        self.__spinner = None

        layout = QHBoxLayout()
        layout.setMargin(0)
        self.setLayout(layout)

        self.__checkbox = QCheckBox(title)
        self.__checkbox.setChecked(False)
        layout.addWidget(self.__checkbox)

        layout.addStretch()

        if time_spinner:
            self.__spinner = self.createTimeSpinner(min_value)
        else:
            self.__spinner = self.createDoubleSpinner(minimum, maximum)

        layout.addWidget(self.__spinner)

        self.__checkbox.stateChanged.connect(self.toggleCheckbox)
        self.setLayout(layout)

    def createDoubleSpinner(self, minimum, maximum):
        spinner = QDoubleSpinBox()
        spinner.setEnabled(False)
        spinner.setMinimumWidth(75)
        spinner.setRange(minimum, maximum)
        spinner.setKeyboardTracking(False)
        spinner.editingFinished.connect(self.spinning)
        spinner.valueChanged.connect(self.spinning)
        return spinner

    def getSpinnerValue(self):
        if self.__time_spinner:
            if self.__spinner is not None:
                index = self.__spinner.value()
                return self.__time_map[index]
            else:
                return None
        else:
            return self.__spinner.value()


    def signalUpdatedValues(self):
        scale_values = {"value": self.getSpinnerValue(), "type_key":self.__type_key, "enabled":self.__checkbox.isChecked()}
        self.plotScaleChanged.emit(scale_values)

    def spinning(self):
        self.signalUpdatedValues()

    def toggleCheckbox(self):
        checked = self.__checkbox.isChecked()
        self.__spinner.setEnabled(checked)
        self.signalUpdatedValues()

    def setValues(self, values):
        self.blockSignals(True)
        type_key = values["type_key"]
        checked = values["enabled"]
        value = values["value"]
        if type_key == self.__type_key:
            self.__checkbox.setChecked(checked)
            if value is not None:
                if self.__time_spinner:
                    index = self.__time_index_map[value]
                    self.__spinner.setValue(index)
                else:
                    self.__spinner.setValue(value)
        self.blockSignals(False)

    def createTimeSpinner(self, min_value):
        def converter(item):
            return "%s" % (str(item.date()))

        spinner = ListSpinBox(self.__time_map)
        spinner.setEnabled(False)
        spinner.setMinimumWidth(75)
        spinner.valueChanged[int].connect(self.spinning)
        spinner.editingFinished.connect(self.spinning)
        spinner.setStringConverter(converter)
        if min_value:
            spinner.setValue(0)
        return spinner