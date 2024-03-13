from __future__ import annotations

import dataclasses
import json
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Optional
from uuid import UUID

import numpy as np
import xarray as xr
import xtgeo
from pydantic import BaseModel

from ert.config import (
    ExtParamConfig,
    Field,
    GenDataConfig,
    GenKwConfig,
    SummaryConfig,
    SurfaceConfig,
)
from ert.config.parsing.context_values import ContextBoolEncoder
from ert.config.response_config import ResponseConfig
from ert.storage.mode import BaseMode, Mode, require_write

if TYPE_CHECKING:
    from ert.config.parameter_config import ParameterConfig
    from ert.run_models.run_arguments import (
        RunArgumentsType,
    )
    from ert.storage.local_ensemble import LocalEnsemble
    from ert.storage.local_storage import LocalStorage

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


class _Index(BaseModel):
    id: UUID
    name: str


class LocalExperiment(BaseMode):
    _parameter_file = Path("parameter.json")
    _responses_file = Path("responses.json")
    _metadata_file = Path("metadata.json")

    def __init__(
        self,
        storage: LocalStorage,
        path: Path,
        mode: Mode,
    ) -> None:
        super().__init__(mode)
        self._storage = storage
        self._path = path
        self._index = _Index.model_validate_json(
            (path / "index.json").read_text(encoding="utf-8")
        )

    @classmethod
    def create(
        cls,
        storage: LocalStorage,
        uuid: UUID,
        path: Path,
        *,
        parameters: Optional[List[ParameterConfig]] = None,
        responses: Optional[List[ResponseConfig]] = None,
        observations: Optional[Dict[str, xr.Dataset]] = None,
        simulation_arguments: Optional[RunArgumentsType] = None,
        name: Optional[str] = None,
    ) -> LocalExperiment:
        if name is None:
            name = datetime.today().strftime("%Y-%m-%d")

        (path / "index.json").write_text(_Index(id=uuid, name=name).model_dump_json())

        parameter_data = {}
        for parameter in parameters or []:
            parameter.save_experiment_data(path)
            parameter_data.update({parameter.name: parameter.to_dict()})
        with open(path / cls._parameter_file, "w", encoding="utf-8") as f:
            json.dump(parameter_data, f, indent=2)

        response_data = {}
        for response in responses or []:
            response_data.update({response.name: response.to_dict()})
        with open(path / cls._responses_file, "w", encoding="utf-8") as f:
            json.dump(response_data, f, default=str, indent=2)

        if observations:
            output_path = path / "observations"
            output_path.mkdir()
            for obs_name, dataset in observations.items():
                dataset.to_netcdf(output_path / f"{obs_name}", engine="scipy")

        with open(path / cls._metadata_file, "w", encoding="utf-8") as f:
            simulation_data = (
                dataclasses.asdict(simulation_arguments) if simulation_arguments else {}
            )
            json.dump(simulation_data, f, cls=ContextBoolEncoder)

        return cls(storage, path, Mode.WRITE)

    @require_write
    def create_ensemble(
        self,
        *,
        ensemble_size: int,
        name: str,
        iteration: int = 0,
        prior_ensemble: Optional[LocalEnsemble] = None,
    ) -> LocalEnsemble:
        return self._storage.create_ensemble(
            self,
            ensemble_size=ensemble_size,
            iteration=iteration,
            name=name,
            prior_ensemble=prior_ensemble,
        )

    @property
    def ensembles(self) -> Generator[LocalEnsemble, None, None]:
        yield from (
            ens for ens in self._storage.ensembles if ens.experiment_id == self.id
        )

    @property
    def metadata(self) -> Dict[str, Any]:
        path = self.mount_point / self._metadata_file
        if not path.exists():
            raise ValueError(f"{str(self._metadata_file)} does not exist")
        with open(path, encoding="utf-8", mode="r") as f:
            return json.load(f)

    @property
    def name(self) -> str:
        return self._index.name

    @property
    def id(self) -> UUID:
        return self._index.id

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

    @cached_property
    def parameter_configuration(self) -> Dict[str, ParameterConfig]:
        params = {}
        for data in self.parameter_info.values():
            param_type = data.pop("_ert_kind")
            params[data["name"]] = _KNOWN_PARAMETER_TYPES[param_type](**data)
        return params

    @cached_property
    def response_configuration(self) -> Dict[str, ResponseConfig]:
        params = {}
        for data in self.response_info.values():
            param_type = data.pop("_ert_kind")
            params[data["name"]] = _KNOWN_RESPONSE_TYPES[param_type](**data)
        return params

    @cached_property
    def update_parameters(self) -> List[str]:
        return [p.name for p in self.parameter_configuration.values() if p.update]

    @cached_property
    def observations(self) -> Dict[str, xr.Dataset]:
        observations = sorted(list(self.mount_point.glob("observations/*")))
        return {
            observation.name: xr.open_dataset(observation, engine="scipy")
            for observation in observations
        }
