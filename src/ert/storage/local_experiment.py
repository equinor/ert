from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional
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
from ert.config.response_config import ResponseConfig
from ert.storage.local_ensemble import LocalEnsemble
from ert.storage.mode import BaseMode, Mode, require_write

if TYPE_CHECKING:
    from ert.config.parameter_config import ParameterConfig
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
    """
    Represents an experiment within the local storage system of ERT.

    Manages the experiment's parameters, responses, observations, and simulation
    arguments. Provides methods to create and access associated ensembles.
    """

    _parameter_file = Path("parameter.json")
    _responses_file = Path("responses.json")
    _metadata_file = Path("metadata.json")

    def __init__(
        self,
        storage: LocalStorage,
        path: Path,
        mode: Mode,
    ) -> None:
        """
        Initialize a LocalExperiment instance.

        Parameters
        ----------
        storage : LocalStorage
            The local storage instance where the experiment is stored.
        path : Path
            The file system path to the experiment data.
        mode : Mode
            The access mode for the experiment (read/write).
        """

        super().__init__(mode)
        self._storage = storage
        self._path = path
        self._index = _Index.model_validate_json(
            (path / "index.json").read_text(encoding="utf-8")
        )
        self._ensembles: List[LocalEnsemble] = []
        self._load_ensembles()

    def _load_ensembles(self) -> None:
        ensembles_path = self.mount_point / "ensembles"
        if not ensembles_path.exists():
            self._ensembles = []
            return

        ensembles: List[LocalEnsemble] = []
        for ensemble_path in ensembles_path.iterdir():
            try:
                ensemble = LocalEnsemble(
                    storage=self._storage, path=ensemble_path, mode=self.mode
                )
                if ensemble.experiment_id == self.id:
                    ensembles.append(ensemble)
            except FileNotFoundError:
                continue

        # Sort ensembles by started_at in reverse order
        self._ensembles = sorted(ensembles, key=lambda x: x.started_at, reverse=True)

    @require_write
    def create_ensemble(
        self,
        *,
        ensemble_size: int,
        name: str,
        iteration: int = 0,
        prior_ensemble: Optional[LocalEnsemble] = None,
    ) -> LocalEnsemble:
        """
        Create a new ensemble associated with this experiment.
        Requires ERT to be in write mode.

        Parameters
        ----------
        ensemble_size : int
            The number of realizations in the ensemble.
        name : str
            The name of the ensemble.
        iteration : int
            The iteration index for the ensemble.
        prior_ensemble : LocalEnsemble, optional
            An optional ensemble to use as a prior.

        Returns
        -------
        local_ensemble : LocalEnsemble
            The newly created ensemble instance.
        """
        ensemble = LocalEnsemble(
            storage=self._storage,
            path=self.mount_point / "ensembles" / name,
            mode=Mode.WRITE,
            ensemble_size=ensemble_size,
            experiment_id=self.id,
            iteration=iteration,
            name=name,
            prior_ensemble_id=prior_ensemble.id if prior_ensemble else None,
        )
        self._ensembles.append(ensemble)
        return ensemble

    def get_ensemble(self, uuid: UUID) -> LocalEnsemble:
        """
        Retrieves an ensemble by UUID.

        Parameters
        ----------
        uuid : UUID
            The UUID of the ensemble to retrieve.

        Returns
        local_ensemble : LocalEnsemble
            The ensemble associated with the given UUID.
        """
        return self.ensembles[uuid]

    def get_ensemble_by_name(self, name: str) -> LocalEnsemble:
        """
        Retrieves an ensemble by name.

        Parameters
        ----------
        name : str
            The name of the ensemble to retrieve.

        Returns
        -------
        local_ensemble : LocalEnsemble
            The ensemble associated with the given name.
        """

        for ens in self.ensembles:
            if ens.name == name:
                return ens
        raise KeyError(f"Ensemble with name '{name}' not found")

    @property
    def ensembles(self) -> List[LocalEnsemble]:
        return self._ensembles

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
        """
        Retrieve a geological surface by name.

        Parameters
        ----------
        name : str
            The name of the surface to retrieve.

        Returns
        -------
        surface : RegularSurface
            The geological surface object.
        """

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
        observations = sorted(self.mount_point.glob("observations/*"))
        return {
            observation.name: xr.open_dataset(observation, engine="scipy")
            for observation in observations
        }
