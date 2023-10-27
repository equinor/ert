from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import xarray as xr

if TYPE_CHECKING:
    from ert.storage.local_ensemble import LocalEnsembleReader
    from ert.storage.local_experiment import LocalExperimentReader
    from ert.storage.local_storage import LocalStorageReader


class LocalRealization:
    def __init__(
        self, ensemble: LocalEnsembleReader, index: int, mode: Literal["r", "w"] = "r"
    ) -> None:
        self.ensemble = ensemble
        self.index = index
        self._path = ensemble._path / f"realization-{index}"

        if mode == "w":
            self._path.mkdir(exist_ok=True, parents=True)

    @property
    def experiment(self) -> LocalExperimentReader:
        return self.ensemble.experiment

    @property
    def storage(self) -> LocalStorageReader:
        return self.ensemble._storage

    def load_dataset(self, group: str) -> xr.Dataset:
        try:
            return xr.open_dataset(self._path / f"{group}.nc", engine="scipy")
        except FileNotFoundError as e:
            raise KeyError(
                f"No dataset '{group}' in storage for realization {self.index}"
            ) from e

    def save_parameters(self, group: str, dataset: xr.Dataset) -> None:
        """Saves the provided dataset under a parameter group and realization index

        Args:
            group: Name of the parameter group under which the dataset is to be saved

            realization: Which realization index this group belongs to

            dataset: Dataset to save. It must contain a variable named
                    'values' which will be used when flattening out the
                    parameters into a 1d-vector.
        """
        if "values" not in dataset.variables:
            raise ValueError(
                f"Dataset for parameter group '{group}' "
                f"must contain a 'values' variable"
            )

        dataset.expand_dims(realizations=[self.index]).to_netcdf(
            self._path / f"{group}.nc", engine="scipy"
        )

    def save_response(self, group: str, data: xr.Dataset) -> None:
        if "realization" not in data.dims:
            data = data.expand_dims({"realization": [self.index]})

        data.to_netcdf(self._path / f"{group}.nc", engine="scipy")
