import sys

from qtpy.QtCore import QMargins, Qt
from qtpy.QtWidgets import QWidget, QHBoxLayout, QComboBox, QToolButton

from ert_gui.ertwidgets import addHelpToWidget, ClosableDialog, resourceIcon
from ert_gui.ertwidgets.analysismodulevariablespanel import AnalysisModuleVariablesPanel
from ert_shared.libres_facade import LibresFacade


class AnalysisModuleSelector(QWidget):
    def __init__(
        self,
        facade: LibresFacade,
        iterable: bool = False,
        load_all: bool = False,
        help_link: str = "",
    ):
        self.facade = facade
        QWidget.__init__(self)
        self._iterable = iterable

        addHelpToWidget(self, help_link)

        layout = QHBoxLayout()

        analysis_module_combo = QComboBox()

        self._module_names = facade.get_analysis_module_names(self._iterable)
        if load_all:
            self._module_names += facade.get_analysis_module_names(not self._iterable)

        suffix = {"STD_ENKF": " - Recommended"}
        for module_name in self._module_names:
            analysis_module_combo.addItem(module_name + suffix.get(module_name, ""))

        self._current_module_name = self._getCurrentAnalysisModuleName()
        if self._current_module_name is not None:
            analysis_module_combo.setCurrentIndex(
                self._module_names.index(self._current_module_name)
            )

        analysis_module_combo.currentIndexChanged[int].connect(
            self.analysisModuleChanged
        )

        variables_popup_button = QToolButton()
        variables_popup_button.setIcon(resourceIcon("ide/small/cog_edit.png"))
        variables_popup_button.clicked.connect(self.showVariablesPopup)
        variables_popup_button.setMaximumSize(20, 20)

        layout.addWidget(analysis_module_combo, 0, Qt.AlignLeft)
        layout.addWidget(variables_popup_button, 0, Qt.AlignLeft)
        layout.setContentsMargins(QMargins(0, 0, 0, 0))
        layout.addStretch()

        self.setLayout(layout)

    def analysisModuleChanged(self, index):
        self._current_module_name = self._module_names[index]

    def _getCurrentAnalysisModuleName(self):
        active_name = self.facade.get_analysis_config().activeModuleName()
        modules = self._module_names

        if active_name in modules:
            return active_name
        elif "STD_ENKF" in modules and not self._iterable:
            return "STD_ENKF"
        elif "LIB_IES" in modules and self._iterable:
            return "LIB_IES"
        elif len(modules) > 0:
            return modules[0]

        return None

    def getSelectedAnalysisModuleName(self):
        return self._current_module_name

    def showVariablesPopup(self):
        if self.getSelectedAnalysisModuleName() is not None:
            variable_dialog = AnalysisModuleVariablesPanel(
                self.getSelectedAnalysisModuleName(), self.facade
            )
            dialog = ClosableDialog("Edit variables", variable_dialog, self.parent())

            dialog.exec_()
