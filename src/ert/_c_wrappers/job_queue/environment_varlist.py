from typing import Any, Dict, List, Optional

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
        env_vars: Optional[List[Any]] = None,
        paths: Optional[List[Any]] = None,
    ):
        env_vars = env_vars or []
        paths = paths or []

        c_ptr = self._alloc()
        super().__init__(c_ptr)

        for key, value in env_vars:
            self.setenv(key, value)
        for key, value in paths:
            self.update_path(key, value)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "EnvironmentVarlist":
        return EnvironmentVarlist(
            env_vars=config_dict.get("SETENV", []),
            paths=config_dict.get("UPDATE_PATH", []),
        )

    def setenv(self, key, value):
        self._setenv(key, value)

    def update_path(self, key, value):
        self._update_path(key, value)

    @property
    def varlist(self):
        return _get_varlist(self)

    @property
    def updatelist(self):
        return _get_updatelist(self)

    def __repr__(self) -> str:
        return (
            f"EnvironmentVarlist(varlist={self.varlist},"
            f" updatelist={self.updatelist})"
        )

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, EnvironmentVarlist)
            and _get_varlist(self) == _get_varlist(other)
            and _get_updatelist(self) == _get_updatelist(other)
        )

    def free(self):
        self._free()
