from cwrap import BaseCClass

from res import _lib
from res import ResPrototype


class LocalUpdateStep(BaseCClass):
    TYPE_NAME = "local_updatestep"

    _size = ResPrototype("int   local_updatestep_get_num_ministep(local_updatestep)")
    _free = ResPrototype("void  local_updatestep_free(local_updatestep)")
    _name = ResPrototype("char* local_updatestep_get_name(local_updatestep)")

    def __init__(self, updatestep_key):
        raise NotImplementedError("Class can not be instantiated directly!")

    def __len__(self):
        """@rtype: int"""
        return self._size()

    def __getitem__(self, index):
        """@rtype: LocalMinistep"""
        if not isinstance(index, int):
            raise TypeError("Keys must be ints, not %s" % str(type(index)))
        if index < 0:
            index += len(self)
        if 0 <= index < len(self):
            return _lib.local.local_updatestep.iget_ministep(self, index)
        else:
            raise IndexError("Invalid index, valid range: [0, %d)" % len(self))

    def attachMinistep(self, ministep):
        _lib.local.local_updatestep.add_ministep(self, ministep)

    def name(self):
        return self._name()

    def getName(self):
        """@rtype: str"""
        return self.name()

    def free(self):
        self._free(self)
