#  Copyright (C) 2013  Statoil ASA, Norway.
#
#  The file 'analysis_module_variables_model.py' is part of ERT - Ensemble based Reservoir Tool.
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
from ert.analysis.analysis_module import AnalysisModule
from ert_gui.models import ErtConnector


class AnalysisModuleVariablesModel(ErtConnector):

    def __init__(self):
        super(AnalysisModuleVariablesModel, self).__init__()
        self.__variable_names = {
            "LAMBDA0": {"type": float, "min": -1, "max": None},
            "LAMBDA_REDUCE": {"type": float},
            "LAMBDA_INCREASE": {"type": float},
            "LAMBDA_MIN": {"type": float},
            "USE_PRIOR": {"type": bool},
            "LOG_FILE": {"type": str},
            "CLEAR_LOG": {"type": bool},
            "LAMBDA_RECALCULATE": {"type": bool}
        }


    def getVariableNames(self, analysis_module_name):
        """ @rtype: list of str """
        analysis_module = self.ert().analysisConfig().getModule(analysis_module_name)
        assert isinstance(analysis_module, AnalysisModule)
        items = []
        for name in self.__variable_names:
            if analysis_module.hasVar(name):
                items.append(name)
        return items

    def getVariableValue(self, analysis_module, name):
        """ @rtype: int or float or bool or str """
        pass

    def getVariableType(self,name):
        return self.__variable_names[name]["type"]


    def setVariableValue(self, analysis_module, name, value):
        pass


