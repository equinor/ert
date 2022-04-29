#  Copyright (C) 2013  Equinor ASA, Norway.
#
#  The file 'analysismodulevariablesmodel.py' is part of ERT.
#
#  ERT is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  ERT is distributed in the hope that it will be useful, but WITHOUT ANY
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or
#  FITNESS FOR A PARTICULAR PURPOSE.
#
#  See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
#  for more details.
from typing import List

from ert_shared.libres_facade import LibresFacade
from res.analysis.analysis_module import AnalysisModule


class AnalysisModuleVariablesModel:

    _VARIABLE_NAMES = AnalysisModule.VARIABLE_NAMES

    @classmethod
    def getVariableNames(
        cls, facade: LibresFacade, analysis_module_name: str
    ) -> List[str]:
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
        cls, facade: LibresFacade, analysis_module_name: str, name: str
    ):
        """@rtype: int or float or bool or str"""
        analysis_module = facade.get_analysis_module(analysis_module_name)
        return analysis_module.getVariableValue(name)
