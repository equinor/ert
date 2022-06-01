from ._base import BaseStorage, Namespace
import numpy as np
import numpy.typing as npt
import xarray as xr
from typing import Optional, List


class XArrayNetCDF(BaseStorage[xr.DataArray]):
    def __init__(self, args: Namespace, keep: bool) -> None:
        super().__init__(args, keep)

        self._engine = "h5netcdf"

    def save_parameter(self, name: str, array: xr.DataArray) -> None:
        da = xr.Dataset({name: array})
        da.to_netcdf(self.path / "params.nc", engine=self._engine, mode="a")

    def save_response(self, name: str, array: xr.DataArray, iens: int) -> None:
        da = xr.Dataset({name: array})
        da.to_netcdf(self.path / f"real_{iens}.nc", mode="a", engine=self._engine)

    def load_response(self, name: str, iens: Optional[List[int]]) -> xr.DataArray:
        if iens is None:
            return xr.concat(
                [xr.open_dataarray(self.path / f"real_{i}.nc", engine=self._engine)
                for i in range(self.args.ensemble_size)]
            )

    def from_numpy(self, array: npt.NDArray[np.float64]) -> xr.DataArray:
        return xr.DataArray(array)
