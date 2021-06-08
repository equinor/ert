#  Copyright (C) 2012  Equinor ASA, Norway.
#
#  The file 'field_config.py' is part of ERT - Ensemble based Reservoir Tool.
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
from res.enkf import LoadFailTypeEnum


class SummaryConfig(BaseCClass):
    TYPE_NAME = "summary_config"
    _alloc = ResPrototype(
        "void* summary_config_alloc(char*, load_fail_type)", bind=False
    )
    _free = ResPrototype("void  summary_config_free(summary_config)")
    _get_var = ResPrototype("char* summary_config_get_var(summary_config)")

    def __init__(self, key, load_fail=LoadFailTypeEnum.LOAD_FAIL_WARN):
        c_ptr = self._alloc(key, load_fail)
        super(SummaryConfig, self).__init__(c_ptr)

    def __repr__(self):
        return "SummaryConfig() %s" % self._ad_str()

    def free(self):
        self._free()

    @property
    def key(self):
        return self._get_var()

    def __ne__(self, other):
        return not self == other

    def __eq__(self, other):
        """@rtype: bool"""
        if self.key != other.key:
            return False

        return True
