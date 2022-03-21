#  Copyright (C) 2013  Equinor ASA, Norway.
#
#  The file 'analysismodulevariablesmodel.py' is part of ERT - Ensemble based Reservoir Tool.
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
import logging
from ert_shared.libres_facade import LibresFacade
from res.analysis.analysis_module import AnalysisModule


class AnalysisModuleVariablesModel(object):

    _VARIABLE_NAMES = {
        "IES_MAX_STEPLENGTH": {
            "type": float,
            "min": 0.1,
            "max": 1.00,
            "step": 0.1,
            "labelname": "Gauss Newton Maximum Steplength",
        },
        "IES_MIN_STEPLENGTH": {
            "type": float,
            "min": 0.1,
            "max": 1.00,
            "step": 0.1,
            "labelname": "Gauss Newton Minimum Steplength",
        },
        "IES_DEC_STEPLENGTH": {
            "type": float,
            "min": 1.1,
            "max": 10.00,
            "step": 0.1,
            "labelname": "Gauss Newton Steplength Decline",
        },
        "IES_INVERSION": {
            "type": int,
            "min": 0,
            "max": 3,
            "step": 1,
            "labelname": "Inversion algorithm",
        },
        "IES_AAPROJECTION": {
            "type": bool,
            "labelname": "Include AA projection",
        },
        "ENKF_TRUNCATION": {
            "type": float,
            "min": -2.0,
            "max": 1,
            "step": 0.01,
            "labelname": "Singular value truncation",
        },
        "ENKF_SUBSPACE_DIMENSION": {
            "type": int,
            "min": -2,
            "max": 2147483647,
            "step": 1,
            "labelname": "Number of singular values",
        },
        "ENKF_NCOMP": {
            "type": int,
            "min": -2,
            "max": 2147483647,
            "step": 1,
            "labelname": "Number of singular values",
        },
    }

    @classmethod
    def getVariableNames(cls, analysis_module: AnalysisModule):
        """@rtype: list of str"""
        assert isinstance(analysis_module, AnalysisModule)
        items = []
        for name in cls._VARIABLE_NAMES:
            if analysis_module.hasVar(name):
                items.append(name)
        return items

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
        result = analysis_module.setVar(name, str(value))

    @classmethod
    def getVariableValue(
        cls, facade: LibresFacade, analysis_module_name: str, name: str
    ):
        """@rtype: int or float or bool or str"""
        logger = logging.getLogger(__name__)
        analysis_module = facade.get_analysis_module(analysis_module_name)
        variable_type = cls.getVariableType(name)
        if variable_type == float:
            return analysis_module.getDouble(name)
        elif variable_type == bool:
            return analysis_module.getBool(name)
        elif variable_type == int:
            return analysis_module.getInt(name)
        else:
            logger.error(f"Unknown variable: {name} of type: {variable_type}")
