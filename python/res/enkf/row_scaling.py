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

    def assign(self, target_size, func):
        """Assign tapering value for all elements.

        The assign() method is the main function used to assign a row scaling
        value to be used as tapering in the update. The first argument is the
        number of elements in the target parameter, and the second argument is
        a callable which will be called with element index as argument.

        In the example below we will assume that tapering is for a field
        variable and we will scale the update with the function exp( -r/r0 )
        where r is the distance from some point and r0 is length scale.

            def sqr(x):
                return x*x

            def exp_decay(grid, pos, r0, data_index):
                x,y,z = grid.get_xyz( active_index = data_index)
                r = math.sqrt( sqr(pos[0] - x) + sqr(pos[1] - y) + sqr(pos[2] - z))
                return math.exp( -r/r0 )


            ens_config = ert.ensembleConfig()
            config_node = ens_config["PORO"]
            field_config = config.node.getFieldModelConfig()
            grid = ert.eclConfig().getGrid()
            pos = grid.get_xyz(ijk=(10,10,1))
            r0 = 10

            if grid.get_num_active() != field_config.get_data_size():
                raise ValuError("Fatal error - inconsistent field size for: {}".format(config_node.getKey())

            # Some local configuration boilerplate has been skipped here.
            local_config = main.getLocalConfig()
            local_data = local_config.createDataset("LOCAL")
            row_scaling = local_data.row_scaling("PORO")
            row_scaling.assign( field_config.get_data_size(), functools.partial(exp_decay, grid, pos, r0))


        In the example below functools.partial() is used to create a callable
        which has access to the necessary context, another alternative would be
        to use a class instance which implements the __call__() method.

        It is an important point that the assign() method does not have any
        context for error checking, i.e. if you call it with an incorrect value
        for the size argument things will silently pass initially but might
        blow up in the subsequent update step.

        """

        for index in range(target_size):
            self[index] = func(index)
