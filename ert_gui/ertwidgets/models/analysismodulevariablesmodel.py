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
from res.analysis.analysis_module import AnalysisModule
from ert_shared import ERT
from ert_gui.ertwidgets.models.ertmodel import getRealizationCount


class AnalysisModuleVariablesModel(object):

    _VARIABLE_NAMES = {
        "IES_MAX_STEPLENGTH": {
            "type": float,
            "min": 0.1,
            "max": 1.00,
            "step": 0.1,
            "labelname": "Gauss Newton Maximum Steplength",
            "pos": 0,
        },
        "IES_MIN_STEPLENGTH": {
            "type": float,
            "min": 0.1,
            "max": 1.00,
            "step": 0.1,
            "labelname": "Gauss Newton Minimum Steplength",
            "pos": 1,
        },
        "IES_DEC_STEPLENGTH": {
            "type": float,
            "min": 1.1,
            "max": 10.00,
            "step": 0.1,
            "labelname": "Gauss Newton Steplength Decline",
            "pos": 2,
        },
        "IES_INVERSION": {
            "type": int,
            "min": 0,
            "max": 3,
            "step": 1,
            "labelname": "Inversion algorithm",
            "pos": 3,
        },
        "IES_DEBUG": {
            "type": bool,
            "labelname": "Print extensive log for IES",
            "pos": 4,
        },
        "IES_LOGFILE": {"type": str, "labelname": "IES Log File", "pos": 5},
        "IES_AAPROJECTION": {
            "type": bool,
            "labelname": "Include AA projection",
            "pos": 11,
        },
        "ENKF_TRUNCATION": {
            "type": float,
            "min": -2.0,
            "max": 1,
            "step": 0.01,
            "labelname": "Singular value truncation",
            "pos": 9,
        },
        "ENKF_SUBSPACE_DIMENSION": {
            "type": int,
            "min": -2,
            "max": 2147483647,
            "step": 1,
            "labelname": "Number of singular values",
            "pos": 10,
        },
        "ENKF_NCOMP": {
            "type": int,
            "min": -2,
            "max": 2147483647,
            "step": 1,
            "labelname": "Number of singular values",
            "pos": 10,
        },
    }

    @classmethod
    def getVariableNames(cls, analysis_module_name):
        """@rtype: list of str"""
        analysis_module = ERT.ert.analysisConfig().getModule(analysis_module_name)
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
    def getVariablePosition(cls, name):
        return cls._VARIABLE_NAMES[name]["pos"]

    @classmethod
    def setVariableValue(cls, analysis_module_name, name, value):
        analysis_module = ERT.ert.analysisConfig().getModule(analysis_module_name)
        result = analysis_module.setVar(name, str(value))

    @classmethod
    def getVariableValue(cls, analysis_module_name, name):
        """@rtype: int or float or bool or str"""
        analysis_module = ERT.ert.analysisConfig().getModule(analysis_module_name)
        variable_type = cls.getVariableType(name)
        if variable_type == float:
            return analysis_module.getDouble(name)
        elif variable_type == bool:
            return analysis_module.getBool(name)
        elif variable_type == str:
            return analysis_module.getStr(name)
        elif variable_type == int:
            return analysis_module.getInt(name)
