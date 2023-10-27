from typing import Literal

from qtpy.QtCore import QMargins, Qt
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QHBoxLayout, QToolButton, QWidget

from ert.gui.ertwidgets import ClosableDialog
from ert.gui.ertwidgets.analysismodulevariablespanel import AnalysisModuleVariablesPanel
from ert.libres_facade import LibresFacade


class AnalysisModuleEdit(QWidget):
    def __init__(
        self,
        facade: LibresFacade,
        module_name: Literal["IES_ENKF", "STD_ENKF"] = "STD_ENKF",
    ):
        self.facade = facade
        QWidget.__init__(self)

        layout = QHBoxLayout()
        self._name = module_name

        variables_popup_button = QToolButton()
        variables_popup_button.setIcon(QIcon("img:edit.svg"))
        variables_popup_button.clicked.connect(self.showVariablesPopup)
        variables_popup_button.setMaximumSize(20, 20)

        layout.addWidget(variables_popup_button, 0, Qt.AlignLeft)
        layout.setContentsMargins(QMargins(0, 0, 0, 0))
        layout.addStretch()

        self.setLayout(layout)

    def showVariablesPopup(self):
        analysis_module = self.facade.get_analysis_module(self._name)
        variable_dialog = AnalysisModuleVariablesPanel(analysis_module, self.facade.get_ensemble_size())
        dialog = ClosableDialog("Edit variables", variable_dialog, self.parent())
        dialog.exec_()
