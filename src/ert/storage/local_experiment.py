from __future__ import annotations

import json
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Optional
from uuid import UUID

import numpy as np
import polars
import xtgeo
from pydantic import BaseModel

from ert.config import (
    ExtParamConfig,
    Field,
    GenKwConfig,
    SurfaceConfig,
)
from ert.config.parsing.context_values import ContextBoolEncoder
from ert.config.response_config import ResponseConfig
from ert.storage.mode import BaseMode, Mode, require_write

if TYPE_CHECKING:
    from ert.config.parameter_config import ParameterConfig
    from ert.storage.local_ensemble import LocalEnsemble
    from ert.storage.local_storage import LocalStorage

_KNOWN_PARAMETER_TYPES = {
    GenKwConfig.__name__: GenKwConfig,
    SurfaceConfig.__name__: SurfaceConfig,
    Field.__name__: Field,
    ExtParamConfig.__name__: ExtParamConfig,
}

from ert.config.responses_index import responses_index


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

    @classmethod
    def create(
        cls,
        storage: LocalStorage,
        uuid: UUID,
        path: Path,
        *,
        parameters: Optional[List[ParameterConfig]] = None,
        responses: Optional[List[ResponseConfig]] = None,
        observations: Optional[Dict[str, polars.DataFrame]] = None,
        simulation_arguments: Optional[Dict[Any, Any]] = None,
        name: Optional[str] = None,
    ) -> LocalExperiment:
        """
        Create a new LocalExperiment and store its configuration data.

        Parameters
        ----------
        storage : LocalStorage
            Storage instance for experiment creation.
        uuid : UUID
            Unique identifier for the new experiment.
        path : Path
            File system path for storing experiment data.
        parameters : list of ParameterConfig, optional
            List of parameter configurations.
        responses : list of ResponseConfig, optional
            List of response configurations.
        observations : dict of str: polars.DataFrame, optional
            Observations dictionary.
        simulation_arguments : SimulationArguments, optional
            Simulation arguments for the experiment.
        name : str, optional
            Experiment name. Defaults to current date if None.

        Returns
        -------
        local_experiment : LocalExperiment
            Instance of the newly created experiment.
        """
        if name is None:
            name = datetime.today().strftime("%Y-%m-%d")

        (path / "index.json").write_text(_Index(id=uuid, name=name).model_dump_json())

        parameter_data = {}
        for parameter in parameters or []:
            parameter.save_experiment_data(path)
            parameter_data.update({parameter.name: parameter.to_dict()})
        storage._write_transaction(
            path / cls._parameter_file,
            json.dumps(parameter_data, indent=2).encode("utf-8"),
        )

        response_data = {}
        for response in responses or []:
            response_data.update({response.response_type: response.to_dict()})
        storage._write_transaction(
            path / cls._responses_file,
            json.dumps(response_data, default=str, indent=2).encode("utf-8"),
        )

        if observations:
            output_path = path / "observations"
            output_path.mkdir()
            for response_type, dataset in observations.items():
                storage._to_parquet_transaction(
                    output_path / f"{response_type}", dataset
                )

        simulation_data = simulation_arguments if simulation_arguments else {}
        storage._write_transaction(
            path / cls._metadata_file,
            json.dumps(simulation_data, cls=ContextBoolEncoder).encode("utf-8"),
        )

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
    def metadata(self) -> Dict[str, Any]:
        path = self.mount_point / self._metadata_file
        if not path.exists():
            raise ValueError(f"{self._metadata_file!s} does not exist")
        with open(path, encoding="utf-8", mode="r") as f:
            return json.load(f)

    @property
    def relative_weights(self) -> str:
        return self.metadata.get("weights", "")

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
            raise ValueError(f"{self._parameter_file!s} does not exist")
        with open(path, encoding="utf-8", mode="r") as f:
            info = json.load(f)
        return info

    @property
    def response_info(self) -> Dict[str, Any]:
        info: Dict[str, Any]
        path = self.mount_point / self._responses_file
        if not path.exists():
            raise ValueError(f"{self._responses_file!s} does not exist")
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

    @property
    def response_configuration(self) -> Dict[str, ResponseConfig]:
        responses = {}
        for data in self.response_info.values():
            ert_kind = data.pop("_ert_kind")
            assert ert_kind in responses_index
            response_cls = responses_index[ert_kind]
            response_instance = response_cls(**data)
            responses[response_instance.response_type] = response_instance

        return responses

    @cached_property
    def update_parameters(self) -> List[str]:
        return [p.name for p in self.parameter_configuration.values() if p.update]

    @cached_property
    def observations(self) -> Dict[str, polars.DataFrame]:
        observations = sorted(self.mount_point.glob("observations/*"))
        return {
            observation.name: polars.read_parquet(f"{observation}")
            for observation in observations
        }

    @cached_property
    def observation_keys(self) -> List[str]:
        """
        Gets all \"name\" values for all observations. I.e.,
        the summary keyword, the gen_data observation name etc.
        """
        keys: List[str] = []
        for df in self.observations.values():
            keys.extend(df["observation_key"].unique())

        return sorted(keys)

    @cached_property
    def response_key_to_response_type(self) -> Dict[str, str]:
        mapping = {}
        for config in self.response_configuration.values():
            for key in config.keys:
                if key == "*":
                    continue

                mapping[key] = config.response_type

        return mapping

    def _update_response_keys(
        self, response_type: str, response_keys: List[str]
    ) -> None:
        """
        When a response is saved to storage, it may contain keys
        that are not explicitly declared in the config. Calling this ensures
        that the response config saved in this storage has keys corresponding
        to the actual received responses.
        """
        if not any(
            k for k in response_keys if k not in self.response_key_to_response_type
        ):
            return None

        responses_configuration = self.response_configuration
        if response_type not in responses_configuration:
            raise KeyError(
                f"Response type {response_type} does not exist in current responses.json"
            )

        config = responses_configuration[response_type]

        new_response_keys = set(response_keys) - set(config.keys)

        if new_response_keys:
            config.keys = sorted(set(config.keys).union(set(response_keys)))
            self._storage._write_transaction(
                self._path / self._responses_file,
                json.dumps(
                    {
                        c.response_type: c.to_dict()
                        for c in responses_configuration.values()
                    },
                    default=str,
                    indent=2,
                ).encode("utf-8"),
            )

            del self.response_key_to_response_type
