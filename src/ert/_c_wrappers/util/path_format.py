import os

from cwrap import BaseCClass

from ert._c_wrappers import ResPrototype

# The path_fmt implementation hinges strongly on variable length
# argument lists in C  not clear if/how that maps over to Python,
# this Python class therefore has *very* limited functionality.


class PathFormat(BaseCClass):
    TYPE_NAME = "path_fmt"
    _alloc = ResPrototype("void* path_fmt_alloc_directory_fmt(char*)", bind=False)
    _str = ResPrototype("char* path_fmt_get_fmt(path_fmt)")
    _free = ResPrototype("void path_fmt_free(path_fmt)")

    def __init__(self, path_fmt):
        c_ptr = self._alloc(path_fmt)
        if c_ptr:
            super().__init__(c_ptr)
        else:
            raise ValueError(f'Unable to construct path format "{path_fmt}"')

    @property
    def format_string(self):
        return self._str()

    def __repr__(self):
        return self._create_repr(f"fmt={self.format_string}")

    def free(self):
        self._free()

    def __ne__(self, other):
        return not self == other

    def __eq__(self, other):
        return os.path.realpath(self._str()) == os.path.realpath(other._str())
