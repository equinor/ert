#  Copyright (C) 2020  Equinor ASA, Norway.
#
#  The file 'row_scaling.py' is part of ERT - Ensemble based Reservoir Tool.
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


class RowScaling(BaseCClass):
    TYPE_NAME = "row_scaling"

    _alloc = ResPrototype("void*  row_scaling_alloc()", bind=False)
    _free = ResPrototype("void   row_scaling_free(row_scaling)")
    _size = ResPrototype("int    row_scaling_get_size(row_scaling)")
    _iset = ResPrototype("double row_scaling_iset(row_scaling, int, double)")
    _iget = ResPrototype("double row_scaling_iget(row_scaling, int)")
    _clamp = ResPrototype("double row_scaling_clamp(row_scaling, double)")

    def __init__(self):
        c_ptr = self._alloc()
        super(RowScaling, self).__init__(c_ptr)

    def free(self):
        self._free()

    def __len__(self):
        return self._size()

    def __setitem__(self, index, value):
        self._iset(index, value)

    def __getitem__(self, index):
        if index < len(self):
            return self._iget(index)

        raise IndexError(
            "Index: {} outside valid range [0,{}>".format(index, len(self))
        )

    def clamp(self, value):
        return self._clamp(value)
