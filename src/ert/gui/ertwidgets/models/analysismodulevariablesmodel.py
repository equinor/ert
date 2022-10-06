from typing import TYPE_CHECKING, List, Union

from ert._c_wrappers.analysis.analysis_module import AnalysisModule
from ert.libres_facade import LibresFacade

if TYPE_CHECKING:
    from ert._c_wrappers.analysis.analysis_module import VariableName


class AnalysisModuleVariablesModel:

    _VARIABLE_NAMES = AnalysisModule.VARIABLE_NAMES

    @classmethod
    def getVariableNames(
        cls, facade: LibresFacade, analysis_module_name: str
    ) -> List["VariableName"]:
        analysis_module = facade.get_analysis_module(analysis_module_name)
        return analysis_module.getVariableNames()

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
        analysis_module.setVar(name, str(value))

    @classmethod
    def getVariableValue(
        cls, facade: LibresFacade, analysis_module_name: str, name: "VariableName"
    ) -> Union[int, float, bool]:
        analysis_module = facade.get_analysis_module(analysis_module_name)
        return analysis_module.getVariableValue(name)
