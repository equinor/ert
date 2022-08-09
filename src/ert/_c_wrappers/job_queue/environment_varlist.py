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


from typing import Dict, Optional

from cwrap import BaseCClass

from ert._c_wrappers import ResPrototype
from ert._clib.env_varlist import _get_updatelist, _get_varlist


class EnvironmentVarlist(BaseCClass):
    TYPE_NAME = "env_varlist"

    _alloc = ResPrototype("void* env_varlist_alloc()", bind=False)
    _free = ResPrototype("void env_varlist_free( env_varlist )")
    _setenv = ResPrototype("void env_varlist_setenv(env_varlist, char*, char*)")
    _update_path = ResPrototype(
        "void env_varlist_update_path(env_varlist, char*, char*)"
    )

    def __init__(
        self,
        vars: Optional[Dict[str, str]] = None,
        paths: Optional[Dict[str, str]] = None,
    ):
        if vars is None:
            vars = {}
        if paths is None:
            paths = {}
        c_ptr = self._alloc()
        super().__init__(c_ptr)

        for key, value in vars.items():
            self._setenv(key, value)
        for key, value in paths.items():
            self._update_path(key, value)

    def __repr__(self) -> str:
        return (
            f"EnvironmentVarlist(varlist={_get_varlist(self)},"
            f" updatelist={_get_updatelist(self)})"
        )

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, EnvironmentVarlist)
            and _get_varlist(self) == _get_varlist(other)
            and _get_updatelist(self) == _get_updatelist(other)
        )

    def free(self):
        self._free()
