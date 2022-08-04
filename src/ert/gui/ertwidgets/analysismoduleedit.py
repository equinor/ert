from qtpy.QtCore import QMargins, Qt
from qtpy.QtWidgets import QHBoxLayout, QToolButton, QWidget
from typing_extensions import Literal

from ert.gui.ertwidgets import ClosableDialog, addHelpToWidget, resourceIcon
from ert.gui.ertwidgets.analysismodulevariablespanel import AnalysisModuleVariablesPanel
from ert.libres_facade import LibresFacade


class AnalysisModuleEdit(QWidget):
    def __init__(
        self,
        facade: LibresFacade,
        module_name: Literal["IES_ENKF", "STD_ENKF"] = "STD_ENKF",
        help_link: str = "",
    ):
        self.facade = facade
        QWidget.__init__(self)

        addHelpToWidget(self, help_link)

        layout = QHBoxLayout()
        self._name = module_name

        variables_popup_button = QToolButton()
        variables_popup_button.setIcon(resourceIcon("edit.svg"))
        variables_popup_button.clicked.connect(self.showVariablesPopup)
        variables_popup_button.setMaximumSize(20, 20)

        layout.addWidget(variables_popup_button, 0, Qt.AlignLeft)
        layout.setContentsMargins(QMargins(0, 0, 0, 0))
        layout.addStretch()

        self.setLayout(layout)

    def showVariablesPopup(self):
        variable_dialog = AnalysisModuleVariablesPanel(self._name, self.facade)
        dialog = ClosableDialog("Edit variables", variable_dialog, self.parent())
        dialog.exec_()
