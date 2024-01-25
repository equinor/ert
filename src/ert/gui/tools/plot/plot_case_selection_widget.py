from typing import List

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QPushButton, QScrollArea, QVBoxLayout, QWidget


class CaseSelectionWidget(QWidget):
    caseSelectionChanged = Signal()
    MAXIMUM_SELECTED = 5
    MINIMUM_SELECTED = 1

    def __init__(self, case_names: List[str]):
        QWidget.__init__(self)
        self._cases = case_names

        self.toggle_buttons: List[CaseSelectCheckButton] = []
        layout = QVBoxLayout()
        self.__case_layout = QVBoxLayout()
        self.__case_layout.setSpacing(0)
        self.__case_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        scrollarea = QScrollArea()
        scrollarea.setWidgetResizable(True)
        button_layout_widget = QWidget()
        button_layout_widget.setLayout(self.__case_layout)
        scrollarea.setWidget(button_layout_widget)
        self._addCheckButtons()
        layout.addWidget(scrollarea)
        self.setLayout(layout)

    def getPlotCaseNames(self) -> List[str]:
        return [widget.text() for widget in self.toggle_buttons if widget.isChecked()]

    def _addCheckButtons(self):
        for case in self._cases:
            button = CaseSelectCheckButton(
                text=case,
                checkbutton_group=self.toggle_buttons,
                parent=self,
                min_select=self.MINIMUM_SELECTED,
                max_select=self.MAXIMUM_SELECTED,
            )
            button.checkStateChanged.connect(self.caseSelectionChanged.emit)
            self.__case_layout.insertWidget(0, button)
            button.setMinimumWidth(20)
            self.toggle_buttons.append(button)

        if len(self.toggle_buttons) > 0:
            self.toggle_buttons[-1].setChecked(True)


class CaseSelectCheckButton(QPushButton):
    checkStateChanged = Signal()

    def __init__(
        self,
        text,
        parent,
        checkbutton_group: List["CaseSelectCheckButton"],
        min_select: int,
        max_select: int,
    ):
        super(CaseSelectCheckButton, self).__init__(text=text, parent=parent)
        self._checkbutton_group = checkbutton_group
        self.min_select = min_select
        self.max_select = max_select
        self.setObjectName("case_selector")
        self.setCheckable(True)

    def nextCheckState(self):
        if (self.isChecked() and not self._verifyCanUncheck()) or not (
            self.isChecked() or self._verifyCanCheck()
        ):
            return
        super().nextCheckState()
        self.checkStateChanged.emit()

    def _verifyCanUncheck(self) -> bool:
        return (
            len([x for x in self._checkbutton_group if x.isChecked()]) > self.min_select
        )

    def _verifyCanCheck(self) -> bool:
        return (
            len([x for x in self._checkbutton_group if x.isChecked()])
            <= self.max_select
        )
