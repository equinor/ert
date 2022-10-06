from cwrap import BaseCClass

from ert._c_wrappers import ResPrototype


class ConfigError(BaseCClass):
    TYPE_NAME = "config_error"
    _free = ResPrototype("void config_error_free(config_error)")
    _count = ResPrototype("int config_error_count(config_error)")
    _iget = ResPrototype("char* config_error_iget(config_error, int)")

    def __init__(self):
        raise NotImplementedError("Class can not be instantiated directly!")

    def __getitem__(self, index):
        """@rtype: str"""
        if not isinstance(index, int):
            raise TypeError("Expected an integer")

        size = len(self)
        if index >= size:
            raise IndexError(f"Index out of range: {index} < {size}")

        return self._iget(index)

    def __len__(self):
        """@rtype: int"""
        return self._count()

    def free(self):
        self._free()
