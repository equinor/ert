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

from res import ResPrototype

from .enums import AnalysisModuleOptionsEnum


class AnalysisModule(BaseCClass):
    TYPE_NAME = "analysis_module"

    _alloc = ResPrototype(
        "void* analysis_module_alloc(int, analysis_mode_enum)", bind=False
    )
    _free = ResPrototype("void analysis_module_free(analysis_module)")
    _set_var = ResPrototype(
        "bool analysis_module_set_var(analysis_module, char*, char*)"
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
        "ENKF_TRUNCATION": {"type": float, "description": "Singular value truncation"},
        "ENKF_SUBSPACE_DIMENSION": {
            "type": int,
            "description": "Number of singular values",
        },
        "ENKF_NCOMP": {"type": int, "description": "Number of singular values"},
    }

    def __init__(self, ens_size, name):
        c_ptr = self._alloc(ens_size, name)
        if not c_ptr:
            raise KeyError("Failed to load internal module:%s" % name)

        super().__init__(c_ptr)

    def getVariableNames(self):
        """@rtype: list of str"""
        items = []
        for name in AnalysisModule.VARIABLE_NAMES:
            if self.hasVar(name):
                items.append(name)
        return items

    def getVariableValue(self, name):
        """@rtype: int or float or bool or str"""
        variable_type = self.getVariableType(name)
        if variable_type == float:
            return self.getDouble(name)
        elif variable_type == bool:
            return self.getBool(name)
        elif variable_type == int:
            return self.getInt(name)

    def getVariableType(self, name):
        """:rtype: type"""
        return AnalysisModule.VARIABLE_NAMES[name]["type"]

    def free(self):
        self._free()

    def __repr__(self):
        if not self:
            return repr(None)
        return f"AnalysisModule(name = {self.name()}, ad = {self._ad_str()})"

    def __assertVar(self, var_name):
        if not self.hasVar(var_name):
            raise KeyError("Module does not support key:%s" % var_name)

    def setVar(self, var_name, value):
        self.__assertVar(var_name)
        string_value = str(value)
        return self._set_var(var_name, string_value)

    def getName(self):
        """:rtype: str"""
        return self.name()

    def name(self):
        return self._get_name()

    def checkOption(self, flag: AnalysisModuleOptionsEnum):
        return self._check_option(flag)

    def hasVar(self, var):
        """:rtype: bool"""
        return self._has_var(var)

    def getDouble(self, var):
        """:rtype: float"""
        self.__assertVar(var)
        return self._get_double(var)

    def getInt(self, var):
        """:rtype: int"""
        self.__assertVar(var)
        return self._get_int(var)

    def getBool(self, var):
        """:rtype: bool"""
        self.__assertVar(var)
        return self._get_bool(var)

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

        var_name_local = self.getVariableNames()
        var_name_other = other.getVariableNames()

        if var_name_local != var_name_other:
            return False

        for a in var_name_local:
            if self.getVariableValue(a) != other.getVariableValue(a):
                return False

        return True
