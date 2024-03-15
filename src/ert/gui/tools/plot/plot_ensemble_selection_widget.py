from typing import List

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QPushButton, QScrollArea, QVBoxLayout, QWidget


class EnsembleSelectionWidget(QWidget):
    ensembleSelectionChanged = Signal()
    MAXIMUM_SELECTED = 5
    MINIMUM_SELECTED = 1

    def __init__(self, ensemble_names: List[str]):
        QWidget.__init__(self)
        self._ensembles = ensemble_names

        self.toggle_buttons: List[EnsembleSelectCheckButton] = []
        layout = QVBoxLayout()
        self.__ensemble_layout = QVBoxLayout()
        self.__ensemble_layout.setSpacing(0)
        self.__ensemble_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        scrollarea = QScrollArea()
        scrollarea.setWidgetResizable(True)
        button_layout_widget = QWidget()
        button_layout_widget.setLayout(self.__ensemble_layout)
        scrollarea.setWidget(button_layout_widget)
        self._addCheckButtons()
        layout.addWidget(scrollarea)
        self.setLayout(layout)

    def getPlotEnsembleNames(self) -> List[str]:
        return [widget.text() for widget in self.toggle_buttons if widget.isChecked()]

    def _addCheckButtons(self):
        for ensemble in self._ensembles:
            button = EnsembleSelectCheckButton(
                text=ensemble,
                checkbutton_group=self.toggle_buttons,
                parent=self,
                min_select=self.MINIMUM_SELECTED,
                max_select=self.MAXIMUM_SELECTED,
            )
            button.checkStateChanged.connect(self.ensembleSelectionChanged.emit)
            self.__ensemble_layout.insertWidget(0, button)
            button.setMinimumWidth(20)
            self.toggle_buttons.append(button)

        if len(self.toggle_buttons) > 0:
            self.toggle_buttons[-1].setChecked(True)


class EnsembleSelectCheckButton(QPushButton):
    checkStateChanged = Signal()

    def __init__(
        self,
        text,
        parent,
        checkbutton_group: List["EnsembleSelectCheckButton"],
        min_select: int,
        max_select: int,
    ):
        super(EnsembleSelectCheckButton, self).__init__(text=text, parent=parent)
        self._checkbutton_group = checkbutton_group
        self.min_select = min_select
        self.max_select = max_select
        self.setObjectName("ensemble_selector")
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
