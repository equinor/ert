#  Copyright (C) 2017  Equinor ASA, Norway.
#
#  This file is part of ERT - Ensemble based Reservoir Tool.
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


class EnvironmentVarlist(BaseCClass):
    TYPE_NAME = "env_varlist"

    _alloc = ResPrototype("void* env_varlist_alloc()", bind=False)
    _free = ResPrototype("void env_varlist_free( env_varlist )")
    _setenv = ResPrototype("void env_varlist_setenv(env_varlist, char*, char*)")
    _get_size = ResPrototype("int env_varlist_get_size(env_varlist)")

    def __init__(self):
        c_ptr = self._alloc()
        super(EnvironmentVarlist, self).__init__(c_ptr)

    def __len__(self):
        """
        Returns the number of elements. Implements len()
        """
        return self._get_size()

    def __setitem__(self, var, value):
        self._setenv(var, value)

    def free(self):
        self._free()
