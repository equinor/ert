from ._base import BaseStorage, Namespace
import numpy as np
import numpy.typing as npt
import xarray as xr
from typing import Optional, Sequence


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

    def load_parameter(self, name: str) -> xr.DataArray:
        return xr.open_dataset(self.path / "params.nc")[name]

    def load_response(self, name: str, iens: Optional[Sequence[int]]) -> xr.DataArray:
        if iens is None:
            iens = range(self.args.ensemble_size)

        return xr.combine_first(
            [xr.open_dataset(self.path / f"real_{i}.nc")[name] for i in iens],
        )

    def from_numpy(self, array: npt.NDArray[np.float64]) -> xr.DataArray:
        return xr.DataArray(array)

    def to_numpy(self, array: xr.DataArray) -> npt.NDArray[np.float64]:
        return array.to_numpy()
