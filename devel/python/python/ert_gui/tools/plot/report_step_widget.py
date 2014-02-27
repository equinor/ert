from PyQt4.QtCore import pyqtSignal
from PyQt4.QtGui import QWidget, QHBoxLayout, QLabel
from ert_gui.models.connectors.plot.report_steps import ReportStepsModel
from ert_gui.widgets.list_spin_box import ListSpinBox
from ert.util.ctime import ctime


class ReportStepWidget(QWidget):
    reportStepTimeSelected = pyqtSignal(int)
    def __init__(self):
        QWidget.__init__(self)


        layout = QHBoxLayout()
        layout.setMargin(0)
        self.setLayout(layout)

        label = QLabel("Report step:")
        layout.addWidget(label)

        layout.addStretch()

        def converter(item):
            return "%s" % (str(item.date()))

        self.__items = ReportStepsModel().getList()
        self.__time_spinner = ListSpinBox(self.__items)
        self.__time_spinner.valueChanged[int].connect(self.valueSelected)
        self.__time_spinner.setStringConverter(converter)
        layout.addWidget(self.__time_spinner)


    def valueSelected(self, index):
        self.reportStepTimeSelected.emit(self.__items[index])

    def getSelectedValue(self):
        """ @rtype: ctime """
        index = self.__time_spinner.value()
        return self.__items[index]


