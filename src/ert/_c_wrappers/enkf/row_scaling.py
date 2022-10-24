from ert._clib.local.row_scaling import RowScaling


def assign(self, target_size, func):
    """Assign tapering value for all elements.

    The assign() method is the main function used to assign a row scaling
    value to be used as tapering in the update. The first argument is the
    number of elements in the target parameter, and the second argument is
    a callable which will be called with element index as argument.

    In the example below we will assume that tapering is for a field
    variable and we will scale the update with the function exp( -r/r0 )
    where r is the distance from some point and r0 is length scale.

        >>> def sqr(x):
        ...     return x*x

        >>> def exp_decay(grid, pos, r0, data_index):
        ...     x,y,z = grid.get_xyz( active_index = data_index)
        ...     r = math.sqrt( sqr(pos[0] - x) + sqr(pos[1] - y) + sqr(pos[2] - z))
        ...     return math.exp( -r/r0 )


        >>> ens_config = ert.ensembleConfig()
        >>> config_node = ens_config["PORO"]
        >>> field_config = config.node.getFieldModelConfig()
        >>> grid = ert.ens_config.grid
        >>> pos = grid.get_xyz(ijk=(10,10,1))
        >>> r0 = 10

        >>> if grid.get_num_active() != field_config.get_data_size():
        ...     raise ValuError(
        ...         "Fatal error - inconsistent field size for: {}".format(
        ...             config_node.getKey()
        ...         )
        ...     )

        >>> # Some local configuration boilerplate has been skipped here.
        >>> update_configuration = main.getLocalConfig()
        >>> local_data = update_configuration.createDataset("LOCAL")
        >>> row_scaling = local_data.row_scaling("PORO")
        >>> row_scaling.assign(
        ...     field_config.get_data_size(),
        ...     functools.partial(exp_decay, grid, pos, r0)
        ... )


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


RowScaling.assign = assign
del assign

__all__ = ["RowScaling"]
