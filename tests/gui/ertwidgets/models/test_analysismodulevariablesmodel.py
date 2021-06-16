from ert_gui.ertwidgets.models.analysismodulevariablesmodel import (
    AnalysisModuleVariablesModel,
)


QT_MAX_SIGNED_INT = 2147483647


def test_getVariableMaximumValue():
    for entry in AnalysisModuleVariablesModel._VARIABLE_NAMES.items():
        if "max" in entry[1] and entry[0] != "CV_NFOLDS":
            assert (
                AnalysisModuleVariablesModel.getVariableMaximumValue(entry[0])
                <= QT_MAX_SIGNED_INT
            ), f"The max value for {entry[0]} is larger than what is allowed for a 32bit signed int."
