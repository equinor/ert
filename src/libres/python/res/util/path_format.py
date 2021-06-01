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
import os

# The path_fmt implementation hinges strongly on variable length
# argument lists in C  not clear if/how that maps over to Python,
# this Python class therefor has *very* limited functionality.


class PathFormat(BaseCClass):
    TYPE_NAME = "path_fmt"
    _alloc = ResPrototype("void* path_fmt_alloc_directory_fmt(char*)", bind=False)
    _str = ResPrototype("char* path_fmt_get_fmt(path_fmt)")
    _free = ResPrototype("void path_fmt_free(path_fmt)")

    def __init__(self, path_fmt):
        c_ptr = self._alloc(path_fmt)
        if c_ptr:
            super(PathFormat, self).__init__(c_ptr)
        else:
            raise ValueError('Unable to construct path format "%s"' % path_fmt)

    def __repr__(self):
        return self._create_repr("fmt=%s" % self._str())

    def free(self):
        self._free()

    def __ne__(self, other):
        return not self == other

    def __eq__(self, other):
        return os.path.realpath(self._str()) == os.path.realpath(other._str())
