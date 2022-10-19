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
        env_vars: Optional[Dict[str, str]] = None,
        paths: Optional[Dict[str, str]] = None,
    ):
        if env_vars is None:
            env_vars = {}
        if paths is None:
            paths = {}
        c_ptr = self._alloc()
        super().__init__(c_ptr)

        for key, value in env_vars.items():
            self.setenv(key, value)
        for key, value in paths.items():
            self.update_path(key, value)

    def setenv(self, key, value):
        self._setenv(key, value)

    def update_path(self, key, value):
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
