from cwrap import BaseCClass

from ert._c_wrappers import ResPrototype


class ConfigPathElm(BaseCClass):
    TYPE_NAME = "config_path_elm"

    _free = ResPrototype("void config_path_elm_free(config_path_elm)")
    _abs_path = ResPrototype("char* config_path_elm_get_abspath(config_path_elm)")

    def __init__(self):
        raise NotImplementedError("Not possible to instantiate ConfigPathElm directly.")

    def free(self):
        self._free()

    @property
    def abs_path(self):
        return self._abs_path()
