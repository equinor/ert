import xarray as xr


def ensure_correct_coordinate_order(ds: xr.Dataset) -> xr.Dataset:
    """
    Ensures correct coordinate order or response/param dataset.
    Slightly less performant than not doing it, but ensure the
    correct coordinate order is applied when doing .to_dataframe().
    It is possible to omit using this and instead pass in the correct
    dim order when doing .to_dataframe(), which is always the same as
    the .dims of the first data var of this dataset.
    """
    # Just to make the order right when
    # doing .to_dataframe()
    # (it seems notoriously hard to tell xarray to just reorder
    # the dimensions/coordinate labels)
    data_vars = list(ds.data_vars.keys())

    # We assume only data vars with the same dimensions,
    # i.e., (realization, *index) for all of them.
    dim_order_of_first_var = ds[data_vars[0]].dims
    return ds[[*dim_order_of_first_var, *data_vars]].sortby(
        dim_order_of_first_var[0]  # "realization" / "realizations"
    )
