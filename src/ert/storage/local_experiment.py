from __future__ import annotations

import json
import logging
import os
import shutil
from collections.abc import Generator
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, TypedDict, cast
from uuid import UUID

import numpy as np
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

from .local_ensemble import LocalEnsemble

if TYPE_CHECKING:
    from ert.storage.local_storage import LocalStorage


class _Index(BaseModel):
    id: UUID
    name: str
    everest_metadata: _EverestExperimentMetadata | None = None


class _EverestExperimentMetadata(BaseModel):
    model_realizations: list[int]
    model_realization_weights: list[float]


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

logger = logging.getLogger(__name__)


class BatchStorageData:
    def __init__(self, path: Path) -> None:
        self._ensemble_path = path

    @property
    def has_data(self) -> bool:
        return any(
            (self._ensemble_path / f"{df_name}.parquet").exists()
            for df_name in LocalEnsemble.BATCH_DATAFRAMES
        )

    @property
    def has_function_results(self) -> bool:
        return any(
            (self._ensemble_path / f"{df_name}.parquet").exists()
            for df_name in _FunctionResults.__annotations__
        )

    @property
    def has_gradient_results(self) -> bool:
        return any(
            (self._ensemble_path / f"{df_name}.parquet").exists()
            for df_name in _GradientResults.__annotations__
        )

    @staticmethod
    def _read_df_if_exists(path: Path) -> pl.DataFrame | None:
        if path.exists():
            return pl.read_parquet(path)
        return None

    @property
    def realization_controls(self) -> pl.DataFrame | None:
        return self._read_df_if_exists(
            self._ensemble_path / "realization_controls.parquet"
        )

    @property
    def batch_objectives(self) -> pl.DataFrame | None:
        return self._read_df_if_exists(self._ensemble_path / "batch_objectives.parquet")

    @property
    def realization_objectives(self) -> pl.DataFrame | None:
        return self._read_df_if_exists(
            self._ensemble_path / "realization_objectives.parquet"
        )

    @property
    def batch_constraints(self) -> pl.DataFrame | None:
        return self._read_df_if_exists(
            self._ensemble_path / "batch_constraints.parquet"
        )

    @property
    def realization_constraints(self) -> pl.DataFrame | None:
        return self._read_df_if_exists(
            self._ensemble_path / "realization_constraints.parquet"
        )

    @property
    def batch_bound_constraint_violations(self) -> pl.DataFrame | None:
        return self._read_df_if_exists(
            self._ensemble_path / "batch_bound_constraint_violations.parquet"
        )

    @property
    def batch_input_constraint_violations(self) -> pl.DataFrame | None:
        return self._read_df_if_exists(
            self._ensemble_path / "batch_input_constraint_violations.parquet"
        )

    @property
    def batch_output_constraint_violations(self) -> pl.DataFrame | None:
        return self._read_df_if_exists(
            self._ensemble_path / "batch_output_constraint_violations.parquet"
        )

    @property
    def batch_objective_gradient(self) -> pl.DataFrame | None:
        return self._read_df_if_exists(
            self._ensemble_path / "batch_objective_gradient.parquet"
        )

    @property
    def perturbation_objectives(self) -> pl.DataFrame | None:
        return self._read_df_if_exists(
            self._ensemble_path / "perturbation_objectives.parquet"
        )

    @property
    def batch_constraint_gradient(self) -> pl.DataFrame | None:
        return self._read_df_if_exists(
            self._ensemble_path / "batch_constraint_gradient.parquet"
        )

    @property
    def perturbation_constraints(self) -> pl.DataFrame | None:
        return self._read_df_if_exists(
            self._ensemble_path / "perturbation_constraints.parquet"
        )

    @cached_property
    def is_improvement(self) -> bool:
        with open(self._ensemble_path / "batch.json", encoding="utf-8") as f:
            info = json.load(f)

        return bool(info["is_improvement"])

    @cached_property
    def batch_id(self) -> bool:
        with open(self._ensemble_path / "batch.json", encoding="utf-8") as f:
            info = json.load(f)

        return info["batch_id"]

    def write_metadata(self, is_improvement: bool) -> None:
        # Clear the cached prop for the new value to take place
        if "is_improvement" in self.__dict__:
            del self.is_improvement

        with open(self._ensemble_path / "batch.json", encoding="utf-8") as f:
            info = json.load(f)
            info["is_improvement"] = is_improvement

        with open(self._ensemble_path / "batch.json", "w", encoding="utf-8") as f:
            json.dump(
                info,
                f,
            )


class _FunctionResults(TypedDict):
    realization_controls: pl.DataFrame
    batch_objectives: pl.DataFrame
    realization_objectives: pl.DataFrame
    batch_constraints: pl.DataFrame | None
    realization_constraints: pl.DataFrame | None
    batch_bound_constraint_violations: pl.DataFrame | None
    batch_input_constraint_violations: pl.DataFrame | None
    batch_output_constraint_violations: pl.DataFrame | None


class _GradientResults(TypedDict):
    batch_objective_gradient: pl.DataFrame | None
    perturbation_objectives: pl.DataFrame | None
    batch_constraint_gradient: pl.DataFrame | None
    perturbation_constraints: pl.DataFrame | None


class FunctionBatchStorageData(BatchStorageData):
    @property
    def realization_controls(self) -> pl.DataFrame:
        df = super().realization_controls
        assert df is not None
        return df

    @property
    def batch_objectives(self) -> pl.DataFrame:
        df = super().batch_objectives
        assert df is not None
        return df

    @property
    def realization_objectives(self) -> pl.DataFrame:
        df = super().realization_objectives
        assert df is not None
        return df

    def to_dict(self) -> dict[str, Any]:
        return {
            "controls": self.realization_controls.drop(
                "batch_id", "realization", "simulation_id"
            ).to_dicts()[0],
            "objectives": self.batch_objectives.drop(
                "batch_id", "total_objective_value"
            ).to_dicts()[0],
            "total_objective_value": self.batch_objectives[
                "total_objective_value"
            ].item(),
            "realization_objectives": self.realization_objectives.drop(
                "batch_id"
            ).to_dicts(),
        }


class GradientBatchStorageData(BatchStorageData):
    @property
    def perturbation_objectives(self) -> pl.DataFrame:
        df = super().perturbation_objectives
        assert df is not None
        return df

    def to_dict(self) -> dict[str, Any]:
        objective_gradient = (
            self.batch_objective_gradient.drop("batch_id")
            .sort("control_name")
            .to_dicts()
            if self.batch_objective_gradient is not None
            else None
        )

        perturbation_objectives = (
            self.perturbation_objectives.drop("batch_id")
            .sort("realization", "perturbation")
            .to_dicts()
        )
        constraint_gradient_dicts = (
            self.batch_constraint_gradient.drop("batch_id")
            .sort("control_name")
            .to_dicts()
            if self.batch_constraint_gradient is not None
            else None
        )

        perturbation_gradient_dicts = (
            self.perturbation_constraints.drop("batch_id")
            .sort("realization", "perturbation")
            .to_dicts()
            if self.perturbation_constraints is not None
            else None
        )

        return {
            "objective_gradient_values": objective_gradient,
            "perturbation_objectives": perturbation_objectives,
            "constraint_gradient": constraint_gradient_dicts,
            "perturbation_constraints": perturbation_gradient_dicts,
        }


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

    def save_everest_metadata(self, metadata: dict[str, Any]) -> None:
        self._index.everest_metadata = _EverestExperimentMetadata.model_validate(
            metadata
        )
        self._storage._write_transaction(
            self._path / "index.json",
            self._index.model_dump_json(indent=2).encode("utf-8"),
        )

    @property
    def everest_batches(self) -> list[BatchStorageData]:
        return [
            BatchStorageData(ens._path)
            for ens in sorted(self.ensembles, key=lambda ens: ens.iteration)
        ]

    @property
    def everest_batches_with_function_results(
        self,
    ) -> list[FunctionBatchStorageData]:
        return [
            FunctionBatchStorageData(b._ensemble_path)
            for b in self.everest_batches
            if b.has_function_results
        ]

    @property
    def everest_batches_with_gradient_results(
        self,
    ) -> list[GradientBatchStorageData]:
        return [
            GradientBatchStorageData(b._ensemble_path)
            for b in self.everest_batches
            if b.has_gradient_results
        ]

    def on_everest_experiment_finished(self) -> None:
        logger.debug("Storing final results Everest storage")

        # This a somewhat arbitrary threshold, this should be a user choice
        # during visualization:
        CONSTRAINT_TOL = 1e-6

        max_total_objective = -np.inf
        for b in self.everest_batches_with_function_results:
            total_objective = b.batch_objectives["total_objective_value"].item()
            bound_constraint_violation = (
                0.0
                if b.batch_bound_constraint_violations is None
                else (
                    b.batch_bound_constraint_violations.drop("batch_id")
                    .to_numpy()
                    .min()
                    .item()
                )
            )
            input_constraint_violation = (
                0.0
                if b.batch_input_constraint_violations is None
                else (
                    b.batch_input_constraint_violations.drop("batch_id")
                    .to_numpy()
                    .min()
                    .item()
                )
            )
            output_constraint_violation = (
                0.0
                if b.batch_output_constraint_violations is None
                else (
                    b.batch_output_constraint_violations.drop("batch_id")
                    .to_numpy()
                    .min()
                    .item()
                )
            )
            if (
                max(
                    bound_constraint_violation,
                    input_constraint_violation,
                    output_constraint_violation,
                )
                < CONSTRAINT_TOL
                and total_objective > max_total_objective
            ):
                b.write_metadata(is_improvement=True)
                max_total_objective = total_objective

    def export_everest_data(
        self,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        batch_dfs_to_join = {}  # type: ignore
        realization_dfs_to_join = {}  # type: ignore
        perturbation_dfs_to_join = {}  # type: ignore

        batch_ids = [b.batch_id for b in self.everest_batches]
        all_controls = self.parameter_keys

        def _try_append_df(
            batch_id: int,
            df: pl.DataFrame | None,
            target: dict[str, list[pl.DataFrame]],
        ) -> None:
            if df is not None:
                if batch_id not in target:  # type: ignore
                    target[batch.batch_id] = []  # type: ignore
                target[batch_id].append(df)  # type: ignore

        def try_append_batch_dfs(batch_id: int, *dfs: pl.DataFrame | None) -> None:
            for df_ in dfs:
                _try_append_df(batch_id, df_, batch_dfs_to_join)

        def try_append_realization_dfs(
            batch_id: int, *dfs: pl.DataFrame | None
        ) -> None:
            for df_ in dfs:
                _try_append_df(batch_id, df_, realization_dfs_to_join)

        def try_append_perturbation_dfs(
            batch_id: int, *dfs: pl.DataFrame | None
        ) -> None:
            for df_ in dfs:
                _try_append_df(batch_id, df_, perturbation_dfs_to_join)

        def pivot_gradient(df: pl.DataFrame) -> pl.DataFrame:
            pivoted_ = df.pivot(on="control_name", index="batch_id", separator=" wrt ")
            return pivoted_.rename(
                {
                    col: f"grad({col})"
                    for col in pivoted_.columns
                    if col != "batch_id" and col not in all_controls
                }
            )

        for batch in self.everest_batches:
            if not batch.has_data:
                continue

            try_append_perturbation_dfs(
                batch.batch_id,
                batch.perturbation_objectives,
                batch.perturbation_constraints,
            )

            try_append_realization_dfs(
                batch.batch_id,
                batch.realization_objectives,
                batch.realization_controls,
                batch.realization_constraints,
            )

            if batch.batch_objective_gradient is not None:
                try_append_batch_dfs(
                    batch.batch_id, pivot_gradient(batch.batch_objective_gradient)
                )

            if batch.batch_constraint_gradient is not None:
                try_append_batch_dfs(
                    batch.batch_id,
                    pivot_gradient(batch.batch_constraint_gradient),
                )

            try_append_batch_dfs(
                batch.batch_id,
                batch.batch_objectives,
                batch.batch_constraints,
            )

        def _join_by_batch(
            dfs: dict[int, list[pl.DataFrame]], on: list[str]
        ) -> list[pl.DataFrame]:
            """
            Creates one dataframe per batch, with one column per input/output,
            including control, objective, constraint, gradient value.
            """
            dfs_to_concat_ = []
            for batch_id in batch_ids:
                if batch_id not in dfs:
                    continue

                batch_df_ = dfs[batch_id][0]
                for bdf_ in dfs[batch_id][1:]:
                    if set(all_controls).issubset(set(bdf_.columns)) and set(
                        all_controls
                    ).issubset(set(batch_df_.columns)):
                        bdf_ = bdf_.drop(all_controls)

                    batch_df_ = batch_df_.join(
                        bdf_,
                        on=on,
                    )

                dfs_to_concat_.append(batch_df_)

            return dfs_to_concat_

        batch_dfs_to_concat = _join_by_batch(batch_dfs_to_join, on=["batch_id"])
        batch_df = pl.concat(batch_dfs_to_concat, how="diagonal")

        realization_dfs_to_concat = _join_by_batch(
            realization_dfs_to_join, on=["batch_id", "realization", "simulation_id"]
        )
        realization_df = pl.concat(realization_dfs_to_concat, how="diagonal")

        perturbation_dfs_to_concat = _join_by_batch(
            perturbation_dfs_to_join, on=["batch_id", "realization", "perturbation"]
        )
        if perturbation_dfs_to_concat:
            # Perturbations exists, proceed as normal
            perturbation_df = pl.concat(perturbation_dfs_to_concat, how="diagonal")
            pert_real_df = pl.concat([realization_df, perturbation_df], how="diagonal")
        else:
            # Discrete methods never have perturbations,
            # append an empty (i.e., null) column
            pert_real_df = realization_df.with_columns(
                pl.lit(None).alias("perturbation")
            )

        pert_real_df = pert_real_df.select(
            "batch_id",
            "realization",
            "perturbation",
            *list(
                set(pert_real_df.columns) - {"batch_id", "realization", "perturbation"}
            ),
        )

        # Avoid name collisions when joining with simulations
        batch_df_renamed = batch_df.rename(
            {
                col: f"batch_{col}"
                for col in batch_df.columns
                if col != "batch_id" and not col.startswith("grad")
            }
        )
        combined_df = pert_real_df.join(
            batch_df_renamed, on="batch_id", how="full", coalesce=True
        )

        def _sort_df(df: pl.DataFrame, index: list[str]) -> pl.DataFrame:
            sorted_cols = index + sorted(set(df.columns) - set(index))
            df_ = df.select(sorted_cols).sort(by=index)
            return df_

        return (
            _sort_df(
                combined_df,
                ["batch_id", "realization", "simulation_id", "perturbation"],
            ),
            _sort_df(
                pert_real_df,
                [
                    "batch_id",
                    "realization",
                    "perturbation",
                    "simulation_id",
                ],
            ),
            _sort_df(batch_df, ["batch_id", "total_objective_value"]),
        )

    def export_everest_opt_results_to_csv(self) -> Path:
        batches_with_data = ",".join(
            {str(b.batch_id) for b in self.everest_batches if b.has_data}
        )
        full_path = (
            self._path
            / f"experiment_results_batches::{','.join(batches_with_data)}.csv"
        )

        # Find old csv to delete
        existing_csv = next(
            (
                Path(f)
                for f in os.listdir(self._path)
                if f.startswith("experiment_results_batches::")
            ),
            None,
        )

        if (
            existing_csv is not None
            and existing_csv.exists()
            and (self._path / existing_csv) != full_path
        ):
            # New batches are added -> overwrite existing csv
            os.remove(existing_csv)

        if not os.path.exists(full_path):
            combined_df, _, _ = self.export_everest_data()
            combined_df.write_csv(full_path)
        return full_path

    @property
    def everest_objective_functions(self) -> EverestObjectivesConfig:
        objectives_config = self.response_configuration.get("everest_objectives")
        assert objectives_config is not None
        return cast(EverestObjectivesConfig, objectives_config)

    @property
    def everest_output_constraint_keys(self) -> list[str]:
        constraints = self.response_configuration.get("everest_constraints")
        if constraints is None:
            return []

        return constraints.keys

    @property
    def has_everest_data(self) -> bool:
        return any(b.has_data for b in self.everest_batches)
