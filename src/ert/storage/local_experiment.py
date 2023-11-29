from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Optional, Union
from uuid import UUID

import numpy as np
import xarray as xr
import xtgeo

from ert.config import (
    ExtParamConfig,
    Field,
    GenDataConfig,
    GenKwConfig,
    SummaryConfig,
    SurfaceConfig,
)
from ert.config.response_config import ResponseConfig

if TYPE_CHECKING:
    from ert.config.parameter_config import ParameterConfig
    from ert.run_models.run_arguments import (
        EnsembleExperimentRunArguments,
        ESMDARunArguments,
        ESRunArguments,
        SIESRunArguments,
        SingleTestRunArguments,
    )
    from ert.storage.local_ensemble import LocalEnsembleAccessor, LocalEnsembleReader
    from ert.storage.local_storage import LocalStorageAccessor, LocalStorageReader

_KNOWN_PARAMETER_TYPES = {
    GenKwConfig.__name__: GenKwConfig,
    SurfaceConfig.__name__: SurfaceConfig,
    Field.__name__: Field,
    ExtParamConfig.__name__: ExtParamConfig,
}


_KNOWN_RESPONSE_TYPES = {
    SummaryConfig.__name__: SummaryConfig,
    GenDataConfig.__name__: GenDataConfig,
}

logger = logging.getLogger(__name__)


class LocalExperimentReader:
    _parameter_file = Path("parameter.json")
    _responses_file = Path("responses.json")
    _simulation_arguments_file = Path("simulation_arguments.json")

    def __init__(self, storage: LocalStorageReader, uuid: UUID, path: Path) -> None:
        self._storage: LocalStorageReader = storage
        self._id = uuid
        self._path = path

    @property
    def ensembles(self) -> Generator[LocalEnsembleReader, None, None]:
        yield from (
            ens for ens in self._storage.ensembles if ens.experiment_id == self.id
        )

    @property
    def id(self) -> UUID:
        return self._id

    @property
    def mount_point(self) -> Path:
        return self._path

    @property
    def parameter_info(self) -> Dict[str, Any]:
        info: Dict[str, Any]
        path = self.mount_point / self._parameter_file
        if not path.exists():
            raise ValueError(f"{str(self._parameter_file)} does not exist")
        with open(path, encoding="utf-8", mode="r") as f:
            info = json.load(f)
        return info

    @property
    def response_info(self) -> Dict[str, Any]:
        info: Dict[str, Any]
        path = self.mount_point / self._responses_file
        if not path.exists():
            raise ValueError(f"{str(self._responses_file)} does not exist")
        with open(path, encoding="utf-8", mode="r") as f:
            info = json.load(f)
        return info

    def get_surface(self, name: str) -> xtgeo.RegularSurface:
        return xtgeo.surface_from_file(
            str(self.mount_point / f"{name}.irap"),
            fformat="irap_ascii",
            dtype=np.float32,
        )

    @property
    def parameter_configuration(self) -> Dict[str, ParameterConfig]:
        params = {}
        for data in self.parameter_info.values():
            param_type = data.pop("_ert_kind")
            params[data["name"]] = _KNOWN_PARAMETER_TYPES[param_type](**data)
        return params

    @property
    def response_configuration(self) -> Dict[str, ResponseConfig]:
        params = {}
        for data in self.response_info.values():
            param_type = data.pop("_ert_kind")
            params[data["name"]] = _KNOWN_RESPONSE_TYPES[param_type](**data)
        return params

    @property
    def observations(self) -> Dict[str, xr.Dataset]:
        observations = list(self.mount_point.glob("observations/*"))
        return {
            observation.name: xr.open_dataset(observation, engine="scipy")
            for observation in observations
        }


class LocalExperimentAccessor(LocalExperimentReader):
    def __init__(
        self,
        storage: LocalStorageAccessor,
        uuid: UUID,
        path: Path,
        parameters: Optional[List[ParameterConfig]] = None,
        responses: Optional[List[ResponseConfig]] = None,
        observations: Optional[Dict[str, xr.Dataset]] = None,
    ) -> None:
        self._storage: LocalStorageAccessor = storage
        self._id = uuid
        self._path = path

        parameters = [] if parameters is None else parameters
        parameter_file = self.mount_point / self._parameter_file

        parameter_data = (
            json.loads(parameter_file.read_text(encoding="utf-8"))
            if parameter_file.exists()
            else {}
        )

        for parameter in parameters:
            parameter.save_experiment_data(self._path)
            parameter_data.update({parameter.name: parameter.to_dict()})

        with open(parameter_file, "w", encoding="utf-8") as f:
            json.dump(parameter_data, f)

        responses = [] if responses is None else responses
        response_file = self.mount_point / self._responses_file
        response_data = (
            json.loads(response_file.read_text(encoding="utf-8"))
            if response_file.exists()
            else {}
        )

        for response in responses:
            response_data.update({response.name: response.to_dict()})
        with open(response_file, "w", encoding="utf-8") as f:
            json.dump(response_data, f, default=str)

        if observations:
            output_path = self.mount_point / "observations"
            Path.mkdir(output_path, parents=True, exist_ok=True)
            for name, dataset in observations.items():
                dataset.to_netcdf(output_path / f"{name}", engine="scipy")

    def create_ensemble(
        self,
        *,
        ensemble_size: int,
        iteration: int = 0,
        name: str,
        prior_ensemble: Optional[LocalEnsembleReader] = None,
    ) -> LocalEnsembleAccessor:
        return self._storage.create_ensemble(
            self,
            ensemble_size=ensemble_size,
            iteration=iteration,
            name=name,
            prior_ensemble=prior_ensemble,
        )

    def write_simulation_arguments(
        self,
        info: Union[
            SingleTestRunArguments,
            EnsembleExperimentRunArguments,
            ESRunArguments,
            ESMDARunArguments,
            SIESRunArguments,
        ],
    ) -> None:
        with open(
            self.mount_point / self._simulation_arguments_file, "w", encoding="utf-8"
        ) as f:
            try:
                json.dump(dataclasses.asdict(info), f)
            except TypeError:
                logger.error(f"Failed to serialize: {info}")
