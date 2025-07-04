from __future__ import annotations

import json
import shutil
from collections.abc import Generator
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any
from uuid import UUID

import polars as pl
from pydantic import BaseModel, Field, TypeAdapter
from surfio import IrapSurface

from ert.config import (
    EverestConstraintsConfig,
    EverestObjectivesConfig,
    ExtParamConfig,
    GenDataConfig,
    GenKwConfig,
    ParameterConfig,
    ResponseConfig,
    SummaryConfig,
    SurfaceConfig,
)
from ert.config import (
    Field as FieldConfig,
)
from ert.config.parsing.context_values import ContextBoolEncoder
from ert.storage.mode import BaseMode, Mode, require_write

if TYPE_CHECKING:
    from ert.storage.local_ensemble import LocalEnsemble
    from ert.storage.local_storage import LocalStorage


class _Index(BaseModel):
    id: UUID
    name: str


_responses_adapter = TypeAdapter(  # type: ignore
    Annotated[
        GenDataConfig
        | SummaryConfig
        | EverestConstraintsConfig
        | EverestObjectivesConfig,
        Field(discriminator="type"),
    ]
)

_parameters_adapter = TypeAdapter(
    list[
        Annotated[
            (GenKwConfig | SurfaceConfig | FieldConfig | ExtParamConfig),
            Field(discriminator="type"),
        ]
    ]
)


class LocalExperiment(BaseMode):
    """
    Represents an experiment within the local storage system of ERT.

    Manages the experiment's parameters, responses, observations, and simulation
    arguments. Provides methods to create and access associated ensembles.
    """

    _parameter_file = Path("parameter.json")
    _responses_file = Path("responses.json")
    _metadata_file = Path("metadata.json")
    _templates_file = Path("templates.json")

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
        parameters: list[ParameterConfig] | None = None,
        responses: list[ResponseConfig] | None = None,
        observations: dict[str, pl.DataFrame] | None = None,
        simulation_arguments: dict[Any, Any] | None = None,
        name: str | None = None,
        templates: list[tuple[str, str]] | None = None,
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
        templates : list of tuple[str, str], optional
            Run templates for the experiment. Defaults to None.

        Returns
        -------
        local_experiment : LocalExperiment
            Instance of the newly created experiment.
        """
        if name is None:
            name = datetime.today().isoformat()

        (path / "index.json").write_text(
            _Index(id=uuid, name=name).model_dump_json(indent=2)
        )

        parameter_data = {}
        for parameter in parameters or []:
            parameter.save_experiment_data(path)
            parameter_data.update({parameter.name: parameter.model_dump(mode="json")})
        storage._write_transaction(
            path / cls._parameter_file,
            json.dumps(parameter_data, indent=2).encode("utf-8"),
        )

        if templates:
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

        response_data = {}
        for response in responses or []:
            response_data.update(
                {response.response_type: response.model_dump(mode="json")}
            )
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

        simulation_data = simulation_arguments or {}
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

        return self._storage.create_ensemble(
            self,
            ensemble_size=ensemble_size,
            iteration=iteration,
            name=name,
            prior_ensemble=prior_ensemble,
        )

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
    def metadata(self) -> dict[str, Any]:
        path = self.mount_point / self._metadata_file
        if not path.exists():
            raise ValueError(f"{self._metadata_file!s} does not exist")
        with open(path, encoding="utf-8") as f:
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
    def parameter_info(self) -> dict[str, Any]:
        info: dict[str, Any]
        with open(self.mount_point / self._parameter_file, encoding="utf-8") as f:
            info = json.load(f)
        return info

    @property
    def templates_configuration(self) -> list[tuple[str, str]]:
        templates: list[tuple[str, str]] = []
        try:
            with open(self.mount_point / self._templates_file, encoding="utf-8") as f:
                templates = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            pass
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
        info: dict[str, Any]
        with open(self.mount_point / self._responses_file, encoding="utf-8") as f:
            info = json.load(f)
        return info

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
            instance.name: instance
            for instance in _parameters_adapter.validate_python(
                self.parameter_info.values()
            )
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
            responses[response_instance.response_type] = response_instance

        return responses

    @cached_property
    def update_parameters(self) -> list[str]:
        return [p.name for p in self.parameter_configuration.values() if p.update]

    @cached_property
    def observations(self) -> dict[str, pl.DataFrame]:
        observations = sorted(self.mount_point.glob("observations/*"))
        return {
            observation.name: pl.read_parquet(f"{observation}")
            for observation in observations
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
                mapping[key] = config.response_type

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
        self._storage._write_transaction(
            self._path / self._responses_file,
            json.dumps(
                {
                    c.response_type: c.model_dump(mode="json")
                    for c in responses_configuration.values()
                },
                default=str,
                indent=2,
            ).encode("utf-8"),
        )

        if self.response_key_to_response_type is not None:
            del self.response_key_to_response_type

        if self.response_type_to_response_keys is not None:
            del self.response_type_to_response_keys

    @property
    def all_parameters_and_gen_data(self) -> pl.DataFrame | None:
        if not self.ensembles:
            return None

        ensemble_dfs = [
            e.all_parameters_and_gen_data
            for e in self.ensembles
            if e.all_parameters_and_gen_data is not None
        ]

        if not ensemble_dfs:
            return None

        return pl.concat(ensemble_dfs)
