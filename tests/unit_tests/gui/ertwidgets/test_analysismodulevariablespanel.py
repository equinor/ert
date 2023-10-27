from unittest.mock import MagicMock, patch

from ert.gui.ertwidgets.analysismodulevariablespanel import AnalysisModuleVariablesPanel
from ert.gui.ertwidgets.models.analysismodulevariablesmodel import (
    AnalysisModuleVariablesModel,
)


def test_createSpinBox(qtbot):
    with patch.object(AnalysisModuleVariablesPanel, "__init__", lambda x, y: None):
        analysis_module_variables_model = AnalysisModuleVariablesModel

        variable_dialog = AnalysisModuleVariablesPanel(MagicMock())

        for entry in AnalysisModuleVariablesModel._VARIABLE_NAMES.items():
            variable_name = entry[0]
            if int == analysis_module_variables_model.getVariableType(variable_name):
                variable_value = 1
                variable_dialog.createSpinBox(
                    variable_name,
                    variable_value,
                    int,
                    analysis_module_variables_model,
                )
