#  Copyright (C) 2013  Equinor ASA, Norway.
#
#  The file 'analysis_module.py' is part of ERT - Ensemble based Reservoir Tool.
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

from cwrap import BaseCClass
from ecl.util.util.rng import RandomNumberGenerator
from res import ResPrototype
from os import path

import res
from res.util import Matrix


class AnalysisModule(BaseCClass):
    TYPE_NAME = "analysis_module"

    _alloc_external = ResPrototype(
        "void* analysis_module_alloc_external(char*)", bind=False
    )
    _alloc_internal = ResPrototype(
        "void* analysis_module_alloc_internal(char*)", bind=False
    )
    _free = ResPrototype("void analysis_module_free(analysis_module)")
    _get_lib_name = ResPrototype("char* analysis_module_get_lib_name(analysis_module)")
    _get_module_internal = ResPrototype(
        "bool analysis_module_internal(analysis_module)"
    )
    _set_var = ResPrototype(
        "bool analysis_module_set_var(analysis_module, char*, char*)"
    )
    _get_table_name = ResPrototype(
        "char* analysis_module_get_table_name(analysis_module)"
    )
    _get_name = ResPrototype("char* analysis_module_get_name(analysis_module)")
    _check_option = ResPrototype(
        "bool analysis_module_check_option(analysis_module, analysis_module_options_enum)"
    )
    _has_var = ResPrototype("bool analysis_module_has_var(analysis_module, char*)")
    _get_double = ResPrototype(
        "double analysis_module_get_double(analysis_module, char*)"
    )
    _get_int = ResPrototype("int analysis_module_get_int(analysis_module, char*)")
    _get_bool = ResPrototype("bool analysis_module_get_bool(analysis_module, char*)")
    _get_str = ResPrototype("char* analysis_module_get_ptr(analysis_module, char*)")
    _init_update = ResPrototype(
        "void analysis_module_init_update(analysis_module, bool_vector, bool_vector, matrix, matrix, matrix, matrix, matrix, rng)"
    )
    _updateA = ResPrototype(
        "void analysis_module_updateA(analysis_module, matrix, matrix, matrix, matrix, matrix, matrix, void*, rng)"
    )
    _initX = ResPrototype(
        "void analysis_module_initX(analysis_module, matrix, matrix, matrix, matrix, matrix, matrix, matrix, rng)"
    )

    # The VARIABLE_NAMES field is a completly broken special case
    # which only applies to the rml module.
    VARIABLE_NAMES = {
        "IES_MAX_STEPLENGTH": {
            "type": float,
            "description": "Max step Length of Gauss Newton Iteration",
        },
        "IES_MIN_STEPLENGTH": {
            "type": float,
            "description": "Min step Length of Gauss Newton Iteration",
        },
        "IES_DEC_STEPLENGTH": {
            "type": float,
            "description": "Decline of step Length in Gauss Newton Iteration",
        },
        "IES_INVERSION": {"type": int, "description": "Inversion algorithm"},
        "IES_DEBUG": {
            "type": bool,
            "description": "Print extensive log for IES analysis steps",
        },
        "IES_LOGFILE": {"type": str, "description": "IES Log File"},
        "IES_AAPROJECTION": {
            "type": str,
            "description": "Include projection Y (A^+A) for n<N-1",
        },
        "LAMBDA0": {"type": float, "description": "Initial Lambda"},
        "USE_PRIOR": {
            "type": bool,
            "description": "Use both Prior and Observation Variability",
        },
        "LAMBDA_REDUCE": {"type": float, "description": "Lambda Reduction Factor"},
        "LAMBDA_INCREASE": {"type": float, "description": "Lambda Incremental Factor"},
        "LAMBDA_MIN": {"type": float, "description": "Minimum Lambda"},
        "LOG_FILE": {"type": str, "description": "Log File"},
        "CLEAR_LOG": {"type": bool, "description": "Clear Existing Log File"},
        "LAMBDA_RECALCULATE": {
            "type": bool,
            "description": "Recalculate Lambda after each Iteration",
        },
        "ENKF_TRUNCATION": {"type": float, "description": "Singular value truncation"},
        "ENKF_SUBSPACE_DIMENSION": {
            "type": int,
            "description": "Number of singular values",
        },
        "ENKF_NCOMP": {"type": int, "description": "Number of singular values"},
        "CV_NFOLDS": {"type": int, "description": "CV_NFOLDS"},
        "FWD_STEP_R2_LIMIT": {"type": float, "description": "FWD_STEP_R2_LIMIT"},
        "CV_PEN_PRESS": {"type": bool, "description": "CV_PEN_PRESS"},
    }

    def __init__(self, name=None, lib_name=None):
        if name is None and lib_name is None:
            raise ValueError("Must supply exactly one of lib or lib_name")

        if name and lib_name:
            raise ValueError("Must supply exactly one of name or lib_name")

        if lib_name:
            c_ptr = self._alloc_external(lib_name)
        else:
            c_ptr = self._alloc_internal(name)
            if not c_ptr:
                raise KeyError("Failed to load internal module:%s" % name)

        super(AnalysisModule, self).__init__(c_ptr)

    def getVariableNames(self):
        """ @rtype: list of str """
        items = []
        for name in AnalysisModule.VARIABLE_NAMES:
            if self.hasVar(name):
                items.append(name)
        return items

    def getVariableValue(self, name):
        """ @rtype: int or float or bool or str """
        variable_type = self.getVariableType(name)
        if variable_type == float:
            return self.getDouble(name)
        elif variable_type == bool:
            return self.getBool(name)
        elif variable_type == str:
            return self.getStr(name)
        elif variable_type == int:
            return self.getInt(name)

    def getVariableType(self, name):
        """ :rtype: type """
        return AnalysisModule.VARIABLE_NAMES[name]["type"]

    def getVariableDescription(self, name):
        """ :rtype: str """
        return AnalysisModule.VARIABLE_NAMES[name]["description"]

    def getVar(self, name):
        return self.getVariableValue(name)

    def free(self):
        self._free()

    def __repr__(self):
        if not self:
            return repr(None)
        nm = self.name()
        tn = self.getTableName()
        ln = self.getLibName()
        mi = "internal" if self.getInternal() else "external"
        ad = self._ad_str()
        fmt = "AnalysisModule(name = %s, table = %s, lib = %s, %s) %s"
        return fmt % (nm, tn, ln, mi, ad)

    def getLibName(self):
        return self._get_lib_name()

    def getInternal(self):
        return self._get_module_internal()

    def __assertVar(self, var_name):
        if not self.hasVar(var_name):
            raise KeyError("Module does not support key:%s" % var_name)

    def setVar(self, var_name, value):
        self.__assertVar(var_name)
        string_value = str(value)
        return self._set_var(var_name, string_value)

    def getTableName(self):
        return self._get_table_name()

    def getName(self):
        """ :rtype: str """
        return self.name()

    def name(self):
        return self._get_name()

    def checkOption(self, flag):
        return self._check_option(flag)

    def hasVar(self, var):
        """ :rtype: bool """
        return self._has_var(var)

    def getDouble(self, var):
        """ :rtype: float """
        self.__assertVar(var)
        return self._get_double(var)

    def getInt(self, var):
        """ :rtype: int """
        self.__assertVar(var)
        return self._get_int(var)

    def getBool(self, var):
        """ :rtype: bool """
        self.__assertVar(var)
        return self._get_bool(var)

    def getStr(self, var):
        """ :rtype: str """
        self.__assertVar(var)
        return self._get_str(var)

    def initUpdate(self, ens_mask, obs_mask, S, R, dObs, E, D, rng):
        self._init_update(ens_mask, obs_mask, S, R, dObs, E, D, rng)

    def updateA(self, A, S, R, dObs, E, D, rng):
        self._updateA(A, S, R, dObs, E, D, None, rng)

    def initX(self, A, S, R, dObs, E, D, rng):
        X = Matrix(A.columns(), A.columns())
        self._initX(X, A, S, R, dObs, E, D, rng)
        return X

    def __ne__(self, other):
        """
        not equal operator between two modules
        :param other: other module to compare with
        :return: True if different
        """
        return not self == other

    def __eq__(self, other):
        """
        equality operator between two modules
        :param other: other module to compare with
        :return: True if the same
        """

        if self.getName() != other.getName():
            return False
        if self.getLibName() != other.getLibName():
            return False
        if self.getTableName() != other.getTableName():
            return False
        if self.getInternal() != other.getInternal():
            return False

        var_name_local = self.getVariableNames()
        var_name_other = other.getVariableNames()

        if var_name_local != var_name_other:
            return False

        for a in var_name_local:
            if self.getVariableValue(a) != other.getVariableValue(a):
                return False

        return True
