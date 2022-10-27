from typing import List, Union

from ert._c_wrappers.analysis import AnalysisMode
from ert._c_wrappers.analysis.analysis_module import get_mode_variables
from ert.libres_facade import LibresFacade


class AnalysisModuleVariablesModel:

    _VARIABLE_NAMES = get_mode_variables(AnalysisMode.ITERATED_ENSEMBLE_SMOOTHER)

    @classmethod
    def getVariableNames(
        cls, facade: LibresFacade, analysis_module_name: str
    ) -> List[str]:
        analysis_module = facade.get_analysis_module(analysis_module_name)
        return analysis_module.get_variable_names()

    @classmethod
    def getVariableType(cls, name):
        return cls._VARIABLE_NAMES[name]["type"]

    @classmethod
    def getVariableMaximumValue(cls, name):
        return cls._VARIABLE_NAMES[name]["max"]

    @classmethod
    def getVariableMinimumValue(cls, name):
        return cls._VARIABLE_NAMES[name]["min"]

    @classmethod
    def getVariableStepValue(cls, name):
        return cls._VARIABLE_NAMES[name]["step"]

    @classmethod
    def getVariableLabelName(cls, name):
        return cls._VARIABLE_NAMES[name]["labelname"]

    @classmethod
    def setVariableValue(
        cls, facade: LibresFacade, analysis_module_name: str, name: str, value: str
    ):
        analysis_module = facade.get_analysis_module(analysis_module_name)
        analysis_module.set_var(name, value)

    @classmethod
    def getVariableValue(
        cls, facade: LibresFacade, analysis_module_name: str, name: str
    ) -> Union[int, float, bool]:
        analysis_module = facade.get_analysis_module(analysis_module_name)
        return analysis_module.get_variable_value(name)
