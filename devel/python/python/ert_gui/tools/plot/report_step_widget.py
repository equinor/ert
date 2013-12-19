from PyQt4.QtGui import QWidget, QHBoxLayout, QLabel
import time
from ert_gui.models.connectors.plot.report_steps import ReportStepsModel
from ert_gui.widgets.list_spin_box import ListSpinBox


class ReportStepWidget(QWidget):
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

        items = ReportStepsModel().getList()
        time_spinner = ListSpinBox(items)
        time_spinner.setStringConverter(converter)
        # time_spinner.setMinimumWidth(150)
        layout.addWidget(time_spinner)



