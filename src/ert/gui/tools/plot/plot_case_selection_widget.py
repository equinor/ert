from qtpy.QtCore import QSignalMapper, Qt, Signal
from qtpy.QtWidgets import QComboBox, QHBoxLayout, QToolButton, QVBoxLayout, QWidget

from ert.gui.ertwidgets import resourceIcon

from .plot_case_model import PlotCaseModel


class CaseSelectionWidget(QWidget):
    caseSelectionChanged = Signal()

    def __init__(self, case_names):
        QWidget.__init__(self)
        self._cases = case_names
        self.__model = PlotCaseModel(case_names)

        self.__signal_mapper = QSignalMapper(self)
        self.__case_selectors = {}
        self.__case_selectors_order = []

        layout = QVBoxLayout()

        add_button_layout = QHBoxLayout()
        self.__add_case_button = QToolButton()
        self.__add_case_button.setObjectName("add_case_button")
        self.__add_case_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.__add_case_button.setText("Add case to plot")
        self.__add_case_button.setIcon(resourceIcon("add_circle_outlined.svg"))
        self.__add_case_button.clicked.connect(self.addCaseSelector)

        add_button_layout.addStretch()
        add_button_layout.addWidget(self.__add_case_button)
        add_button_layout.addStretch()

        layout.addLayout(add_button_layout)

        self.__case_layout = QVBoxLayout()
        self.__case_layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(self.__case_layout)

        self.addCaseSelector(disabled=True)
        layout.addStretch()

        self.setLayout(layout)

        self.__signal_mapper.mapped[QWidget].connect(self.removeWidget)

    def __caseName(self, widget) -> str:
        return str(self.__case_selectors[widget].currentText())

    def getPlotCaseNames(self):
        if self.__model.rowCount() == 0:
            return []

        return [self.__caseName(widget) for widget in self.__case_selectors_order]

    def checkCaseCount(self):
        state = True
        if len(self.__case_selectors_order) == 5:
            state = False

        self.__add_case_button.setEnabled(state)

    def addCaseSelector(self, disabled=False):
        widget = QWidget()

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        widget.setLayout(layout)

        combo = QComboBox()
        combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(20)
        combo.setModel(self.__model)

        combo.currentIndexChanged.connect(self.caseSelectionChanged.emit)

        layout.addWidget(combo, 1)

        button = QToolButton()
        button.setAutoRaise(True)
        button.setDisabled(disabled)
        button.setIcon(resourceIcon("delete_to_trash.svg"))
        button.clicked.connect(self.__signal_mapper.map)

        layout.addWidget(button)

        self.__case_selectors[widget] = combo
        self.__case_selectors_order.append(widget)
        self.__signal_mapper.setMapping(button, widget)

        self.__case_layout.addWidget(widget)

        self.checkCaseCount()
        self.caseSelectionChanged.emit()

    def removeWidget(self, widget):
        self.__case_layout.removeWidget(widget)
        del self.__case_selectors[widget]
        self.__case_selectors_order.remove(widget)
        widget.setParent(None)
        self.caseSelectionChanged.emit()

        self.checkCaseCount()
