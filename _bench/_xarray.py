from ._base import BaseStorage
import argparse
import numpy as np
import numpy.typing as npt
import xarray as xr


class XArrayNetCDF(BaseStorage[xr.DataArray]):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)

        self._engine = "h5netcdf"

    def save_parameter(self, name: str, array: xr.DataArray) -> None:
        da = xr.Dataset({name: array})
        da.to_netcdf(self.path / "params.nc", engine=self._engine, mode="a")

    def save_response(self, name: str, array: xr.DataArray, iens: int) -> None:
        da = xr.Dataset({name: array})
        da.to_netcdf(self.path / f"real_{iens}.nc", mode="a", engine=self._engine)

    def from_numpy(self, array: npt.NDArray[np.float64]) -> xr.DataArray:
        return xr.DataArray(array)
