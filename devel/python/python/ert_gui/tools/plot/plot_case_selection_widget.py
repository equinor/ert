from PyQt4.QtCore import pyqtSignal
from PyQt4.QtGui import QWidget, QVBoxLayout, QColor, QHBoxLayout, QToolButton, QComboBox, QPushButton
from ert_gui.tools.plot import PlotCaseModel
from ert_gui.widgets import util
from ert_gui.widgets.legend import Legend


class CaseSelectionWidget(QWidget):

    def __init__(self):
        QWidget.__init__(self)

        self.__model = PlotCaseModel()
        self.__colors = [QColor(0, 128, 255), QColor(128, 255, 0), QColor(255, 128, 0), QColor(255, 0, 128)]
        self.__count = 0

        layout = QVBoxLayout()

        add_button_layout = QHBoxLayout()
        button = QPushButton(util.resourceIcon("ide/small/add"), "Add case to plot")
        add_button_layout.addStretch()
        add_button_layout.addWidget(button)
        add_button_layout.addStretch()

        layout.addLayout(add_button_layout)
        layout.addWidget(self.createCaseLayout(disabled=True))
        layout.addWidget(self.createCaseLayout())
        layout.addStretch()

        self.setLayout(layout)




    def createCaseLayout(self, disabled=False):
        widget = QWidget()

        layout = QHBoxLayout()
        layout.setMargin(0)
        widget.setLayout(layout)

        legend = Legend("", self.__colors[self.__count])
        legend.setMinimumWidth(20)
        layout.addWidget(legend)

        combo = QComboBox()
        combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(20)
        combo.setModel(self.__model)

        layout.addWidget(combo)

        button = QToolButton()
        button.setAutoRaise(True)
        button.setDisabled(disabled)
        button.setIcon(util.resourceIcon("ide/small/delete"))
        layout.addStretch()
        layout.addWidget(button)

        self.__count += 1

        return widget



