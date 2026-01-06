from __future__ import annotations

import contextlib
import json
import shutil
from collections.abc import Generator
from datetime import datetime
from enum import StrEnum, auto
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any
from uuid import UUID

import polars as pl
from pydantic import BaseModel, Field, TypeAdapter
from surfio import IrapSurface

from ert.config import (
    EverestControl,
    GenKwConfig,
    KnownResponseTypes,
    ParameterConfig,
    ResponseConfig,
    SurfaceConfig,
)
from ert.config import Field as FieldConfig
from ert.config._create_observation_dataframes import (
    create_observation_dataframes,
)
from ert.config._observations import Observation

from .mode import BaseMode, Mode, require_write

if TYPE_CHECKING:
    from .local_ensemble import LocalEnsemble
    from .local_storage import LocalStorage


class ExperimentState(StrEnum):
    pending = auto()
    running = auto()
    completed = auto()
    stopped = auto()
    failed = auto()
    never_run = auto()


class ExperimentStatus(BaseModel):
    message: str = Field(default="")
    status: ExperimentState = Field(default=ExperimentState.pending)


class _Index(BaseModel):
    id: UUID
    name: str
    # An experiment may point to ensembles that are originated
    # from a different experiment. For example, a manual update
    # is a separate experiment from the one that created the prior.
    ensembles: list[UUID]
    experiment: dict[str, Any] = {}
    status: ExperimentStatus | None = Field(default=None)


_responses_adapter = TypeAdapter(  # type: ignore
    Annotated[
        KnownResponseTypes,
        Field(discriminator="type"),
    ]
)

_parameters_adapter = TypeAdapter(  # type: ignore
    Annotated[
        (GenKwConfig | SurfaceConfig | FieldConfig | EverestControl),
        Field(discriminator="type"),
    ]
)


class LocalExperiment(BaseMode):
    """
    Represents an experiment within the local storage system of ERT.

    Manages the experiment's parameters, responses, observations, and simulation
    arguments. Provides methods to create and access associated ensembles.
    """

    _templates_file = Path("templates.json")
    _index_file = Path("index.json")

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
            (path / self._index_file).read_text(encoding="utf-8")
        )

    @classmethod
    def create(
        cls,
        storage: LocalStorage,
        uuid: UUID,
        path: Path,
        experiment_config: dict[str, Any],
        name: str | None = None,
    ) -> LocalExperiment:
        if name is None:
            name = datetime.today().isoformat()

        storage._write_transaction(
            path / cls._index_file,
            _Index(id=uuid, name=name, ensembles=[], experiment=experiment_config)
            .model_dump_json(indent=2, exclude_none=True)
            .encode("utf-8"),
        )

        templates = experiment_config.get("ert_templates")
        if templates is not None:
            templates_path = path / "templates"
            templates_path.mkdir(parents=True, exist_ok=True)
            templates_abs: list[tuple[str, str]] = []
            for idx, (src, dst) in enumerate(templates):
                incoming_template = Path(src)
                template_file_path = (
                    templates_path
                    / f"{incoming_template.stem}_{idx}{incoming_template.suffix}"
                )
                shutil.copyfile(incoming_template, template_file_path)
                templates_abs.append((str(template_file_path.relative_to(path)), dst))
            storage._write_transaction(
                path / cls._templates_file,
                json.dumps(templates_abs).encode("utf-8"),
            )

        observation_declarations = experiment_config.get("observations")
        if observation_declarations:
            output_path = path / "observations"
            output_path.mkdir(parents=True, exist_ok=True)

            responses_list = experiment_config.get("response_configuration", [])
            rft_config_json = next(
                (r for r in responses_list if r.get("type") == "rft"), None
            )
            rft_config = (
                _responses_adapter.validate_python(rft_config_json)
                if rft_config_json is not None
                else None
            )

            obs_adapter = TypeAdapter(Observation)  # type: ignore
            obs_objs: list[Observation] = []
            for od in observation_declarations:
                obs_objs.append(obs_adapter.validate_python(od))

            datasets = create_observation_dataframes(obs_objs, rft_config)
            for response_type, df in datasets.items():
                storage._to_parquet_transaction(output_path / response_type, df)

        return cls(storage, path, Mode.WRITE)

    @require_write
    def create_ensemble(
        self,
        *,
        ensemble_size: int,
        name: str,
        iteration: int = 0,
        prior_ensemble: LocalEnsemble | None = None,
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

        ensemble = self._storage.create_ensemble(
            self,
            ensemble_size=ensemble_size,
            iteration=iteration,
            name=name,
            prior_ensemble=prior_ensemble,
        )

        self._index.ensembles.append(ensemble.id)
        self._storage._write_transaction(
            self._path / "index.json",
            self._index.model_dump_json(indent=2).encode("utf-8"),
        )
        return ensemble

    @property
    def ensembles(self) -> Generator[LocalEnsemble]:
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
    def relative_weights(self) -> str:
        assert self.experiment_config is not None
        return self.experiment_config.get("weights", "")

    @property
    def name(self) -> str:
        return self._index.name

    @property
    def id(self) -> UUID:
        return self._index.id

    @property
    def status(self) -> ExperimentStatus | None:
        return self._index.status

    @status.setter
    @require_write
    def status(self, status: ExperimentStatus) -> None:
        if status != self._index.status:
            self._index.status = status
            self._storage._write_transaction(
                self.mount_point / self._index_file,
                self._index.model_dump_json(indent=2).encode("utf-8"),
            )

    @property
    def mount_point(self) -> Path:
        return self._path

    @property
    def parameter_info(self) -> dict[str, Any]:
        parameters_list = self.experiment_config.get("parameter_configuration", [])
        return {parameter["name"]: parameter for parameter in parameters_list}

    @property
    def templates_configuration(self) -> list[tuple[str, str]]:
        templates: list[tuple[str, str]] = []
        with contextlib.suppress(FileNotFoundError, json.JSONDecodeError):
            templates = json.loads(
                (self.mount_point / self._templates_file).read_text(encoding="utf-8")
            )
        templates_with_content: list[tuple[str, str]] = []
        for source_file, target_file in templates:
            try:
                file_content = (self.mount_point / source_file).read_text("utf-8")
                templates_with_content.append((file_content, target_file))
            except UnicodeDecodeError as e:
                raise ValueError(
                    f"Unsupported non UTF-8 character found in file: {source_file}"
                ) from e
        return templates_with_content

    @property
    def response_info(self) -> dict[str, Any]:
        responses_list = self.experiment_config.get("response_configuration", [])
        return {response["type"]: response for response in responses_list}

    def get_surface(self, name: str) -> IrapSurface:
        """
        Retrieve a geological surface by name.

        Parameters
        ----------
        name : str
            The name of the surface to retrieve.

        Returns
        -------
        surface : IrapSurface
            The geological surface object.
        """

        return IrapSurface.from_ascii_file(self.mount_point / f"{name}.irap")

    @cached_property
    def parameter_configuration(self) -> dict[str, ParameterConfig]:
        return {
            name: _parameters_adapter.validate_python(cfg)
            for name, cfg in self.parameter_info.items()
        }

    @cached_property
    def parameter_keys(self) -> list[str]:
        keys = []
        for config in self.parameter_configuration.values():
            keys += config.parameter_keys

        return keys

    @cached_property
    def parameter_group_to_parameter_keys(self) -> dict[str, list[str]]:
        return {
            config.name: config.parameter_keys
            for config in self.parameter_configuration.values()
        }

    @cached_property
    def response_key_to_observation_key(self) -> dict[str, dict[str, list[str]]]:
        # response_type->response_key->[observation_keys]
        return {
            response_type: {
                d["response_key"]: d["observation_key"]
                for d in obs_df.group_by("response_key")
                .agg([pl.col("observation_key").drop_nulls().unique()])
                .to_dicts()
            }
            for response_type, obs_df in self.observations.items()
        }

    @property
    def response_configuration(self) -> dict[str, ResponseConfig]:
        responses = {}

        for data in self.response_info.values():
            response_instance = _responses_adapter.validate_python(data)
            responses[response_instance.type] = response_instance

        return responses

    @cached_property
    def update_parameters(self) -> list[str]:
        return [p.name for p in self.parameter_configuration.values() if p.update]

    @cached_property
    def observations(self) -> dict[str, pl.DataFrame]:
        obs_dir = self.mount_point / "observations"

        if obs_dir.exists():
            datasets: dict[str, pl.DataFrame] = {}
            for p in obs_dir.iterdir():
                if not p.is_file():
                    continue
                try:
                    df = pl.read_parquet(p)
                except Exception:
                    continue
                datasets[p.stem] = df
            return datasets

        serialized_observations = self.experiment_config.get("observations", None)
        if not serialized_observations:
            return {}

        output_path = self.mount_point / "observations"
        output_path.mkdir(parents=True, exist_ok=True)

        rft_cfg = None
        try:
            responses_list = self.experiment_config.get("response_configuration", [])
            for r in responses_list:
                if r.get("type") == "rft":
                    rft_cfg = _responses_adapter.validate_python(r)
                    break
        except Exception:
            rft_cfg = None

        obs_adapter = TypeAdapter(Observation)  # type: ignore
        obs_objs: list[Observation] = []
        for od in serialized_observations:
            obs_objs.append(obs_adapter.validate_python(od))

        datasets = create_observation_dataframes(obs_objs, rft_cfg)
        for response_type, df in datasets.items():
            self._storage._to_parquet_transaction(output_path / response_type, df)

        return {
            p.stem: pl.read_parquet(p)
            for p in (self.mount_point / "observations").iterdir()
            if p.is_file()
        }

    @cached_property
    def observation_keys(self) -> list[str]:
        """
        Gets all \"name\" values for all observations. I.e.,
        the summary keyword, the gen_data observation name etc.
        """
        keys: list[str] = []
        for df in self.observations.values():
            keys.extend(df["observation_key"].unique())

        return sorted(keys)

    @cached_property
    def response_key_to_response_type(self) -> dict[str, str]:
        mapping = {}
        for config in self.response_configuration.values():
            for key in config.keys if config.has_finalized_keys else []:
                mapping[key] = config.type

        return mapping

    @cached_property
    def response_type_to_response_keys(self) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}

        for response_key, response_type in self.response_key_to_response_type.items():
            if response_type not in result:
                result[response_type] = []

            result[response_type].append(response_key)

        for keys in result.values():
            keys.sort()

        return result

    def _has_finalized_response_keys(self, response_type: str) -> bool:
        responses_configuration = self.response_configuration
        if response_type not in responses_configuration:
            raise KeyError(
                f"Response type {response_type} does not "
                "exist in current responses.json"
            )

        return responses_configuration[response_type].has_finalized_keys

    def _update_response_keys(
        self, response_type: str, response_keys: list[str]
    ) -> None:
        """
        When a response is saved to storage, it may contain keys
        that are not explicitly declared in the config. Calling this ensures
        that the response config saved in this storage has keys corresponding
        to the actual received responses.
        """
        responses_configuration = self.response_configuration
        if response_type not in responses_configuration:
            raise KeyError(
                f"Response type {response_type} does not "
                "exist in current responses.json"
            )

        config = responses_configuration[response_type]

        config.keys = sorted(response_keys)
        config.has_finalized_keys = True

        response_index = next(
            i
            for i, c in enumerate(self.experiment_config["response_configuration"])
            if c["type"] == response_type
        )
        self.experiment_config["response_configuration"][response_index] = (
            config.model_dump(mode="json")
        )

        self._storage._write_transaction(
            self._path / self._index_file,
            self._index.model_dump_json(indent=2).encode("utf-8"),
        )

        if self.response_key_to_response_type is not None:
            del self.response_key_to_response_type

        if self.response_type_to_response_keys is not None:
            del self.response_type_to_response_keys

    @property
    def experiment_config(self) -> dict[str, Any]:
        return self._index.experiment
