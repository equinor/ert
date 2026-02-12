from __future__ import annotations

import dataclasses
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal, Self

import polars as pl
from polars.datatypes import DataTypeClass
from pydantic import BaseModel, Field, TypeAdapter

from ert.base_model_context import BaseModelWithContextSupport
from ert.config import (
    ESSettings,
    EverestConstraintsConfig,
    EverestControl,
    EverestObjectivesConfig,
    ForwardModelStep,
    GenDataConfig,
    GenKwConfig,
    HookRuntime,
    KnownDerivedResponseTypes,
    KnownResponseTypes,
    ModelConfig,
    Observation,
    ObservationSettings,
    QueueConfig,
    SummaryConfig,
    SurfaceConfig,
    Workflow,
)
from ert.config import Field as FieldConfig
from ert.storage.local_experiment import ExperimentConfig, ExperimentType
from everest.config import InputConstraintConfig, OptimizationConfig
from everest.config import ModelConfig as EverestModelConfig


# https://github.com/pola-rs/polars/issues/13152#issuecomment-1864600078
# PS: Serializing/deserializing schema is scheduled to be added to polars core,
# ref https://github.com/pola-rs/polars/issues/20426
# then this workaround can be omitted.
def str_to_dtype(dtype_str: str) -> pl.DataType:
    dtype = eval(f"pl.{dtype_str}")
    if isinstance(dtype, DataTypeClass):
        dtype = dtype()
    return dtype


class DictEncodedDataFrame(BaseModel):
    type: Literal["dicts"]
    data: list[dict[str, Any]]
    datatypes: dict[str, str]

    @classmethod
    def from_polars(cls, data: pl.DataFrame) -> Self:
        str_schema = {k: str(dtype) for k, dtype in data.schema.items()}
        return cls(type="dicts", data=data.to_dicts(), datatypes=str_schema)

    def to_polars(self) -> pl.DataFrame:
        return pl.from_dicts(
            self.data,
            schema={
                col: str_to_dtype(dtype_str)
                for col, dtype_str in self.datatypes.items()
            },
        )


class RunModelConfig(BaseModelWithContextSupport):
    storage_path: str
    runpath_file: Path
    user_config_file: Path
    env_vars: dict[str, str]
    env_pr_fm_step: dict[str, dict[str, Any]]
    runpath_config: ModelConfig
    queue_config: QueueConfig
    forward_model_steps: list[ForwardModelStep]
    substitutions: dict[str, str]
    hooked_workflows: defaultdict[HookRuntime, list[Workflow]]
    active_realizations: list[bool]
    log_path: Path
    random_seed: int
    start_iteration: int = 0
    minimum_required_realizations: int = 0
    supports_rerunning_failed_realizations: ClassVar[bool] = False


class InitialEnsembleRunModelConfig(RunModelConfig):
    experiment_name: str
    design_matrix: DictEncodedDataFrame | None
    parameter_configuration: list[
        Annotated[
            (GenKwConfig | SurfaceConfig | FieldConfig | EverestControl),
            Field(discriminator="type"),
        ]
    ]
    response_configuration: list[
        Annotated[
            (KnownResponseTypes),
            Field(discriminator="type"),
        ]
    ]
    derived_response_configuration: list[
        Annotated[
            (KnownDerivedResponseTypes),
            Field(discriminator="type"),
        ]
    ]
    ert_templates: list[tuple[str, str]]
    observations: list[Observation] | None = None

    def _initial_ensemble_experiment_config(self) -> ExperimentConfig:
        experiment_config: ExperimentConfig = {
            "ert_templates": self.ert_templates,
            "parameter_configuration": [
                param.model_dump(mode="json") for param in self.parameter_configuration
            ],
            "response_configuration": [
                resp.model_dump(mode="json") for resp in self.response_configuration
            ],
            "derived_response_configuration": [
                resp.model_dump(mode="json")
                for resp in self.derived_response_configuration
            ],
        }

        if self.observations is not None:
            experiment_config["observations"] = [
                obs.model_dump(mode="json") for obs in self.observations
            ]

        if self.design_matrix is not None:
            experiment_config["design_matrix"] = self.design_matrix.model_dump(
                mode="json"
            )

        return experiment_config


class UpdateRunModelConfig(RunModelConfig):
    target_ensemble: str
    analysis_settings: ESSettings
    update_settings: ObservationSettings

    def _update_experiment_config(self) -> ExperimentConfig:
        return {
            "target_ensemble": self.target_ensemble,
            "analysis_settings": self.analysis_settings.model_dump(mode="json"),
            "update_settings": dataclasses.asdict(self.update_settings),
        }


class EnsembleSmootherConfig(InitialEnsembleRunModelConfig, UpdateRunModelConfig):
    experiment_type: Literal["Ensemble Smoother"] = (
        ExperimentType.ENSEMBLE_SMOOTHER.value
    )

    def to_experiment_config(self) -> ExperimentConfig:
        return {
            "experiment_type": ExperimentType.ENSEMBLE_SMOOTHER.value,
            **self._initial_ensemble_experiment_config(),
            **self._update_experiment_config(),
        }


class EnsembleInformationFilterConfig(
    InitialEnsembleRunModelConfig, UpdateRunModelConfig
):
    experiment_type: Literal["Ensemble Information Filter"] = (
        ExperimentType.ENSEMBLE_INFORMATION_FILTER.value
    )

    def to_experiment_config(self) -> ExperimentConfig:
        return {
            "experiment_type": ExperimentType.ENSEMBLE_INFORMATION_FILTER.value,
            **self._initial_ensemble_experiment_config(),
            **self._update_experiment_config(),
        }


class EnsembleExperimentConfig(InitialEnsembleRunModelConfig):
    experiment_type: Literal["Ensemble Experiment"] = (
        ExperimentType.ENSEMBLE_EXPERIMENT.value
    )
    target_ensemble: str
    supports_rerunning_failed_realizations: ClassVar[bool] = True

    def to_experiment_config(self) -> ExperimentConfig:
        return {
            "experiment_type": ExperimentType.ENSEMBLE_EXPERIMENT.value,
            **self._initial_ensemble_experiment_config(),
            "target_ensemble": self.target_ensemble,
        }


class EvaluateEnsembleConfig(RunModelConfig):
    experiment_type: Literal["Evaluate Ensemble"] = (
        ExperimentType.EVALUATE_ENSEMBLE.value
    )
    ensemble_id: str
    supports_rerunning_failed_realizations: ClassVar[bool] = True


EverestResponseTypes = (
    EverestObjectivesConfig | EverestConstraintsConfig | SummaryConfig | GenDataConfig
)
EverestResponseTypesAdapter = TypeAdapter(  # type: ignore
    Annotated[
        EverestResponseTypes,
        Field(discriminator="type"),
    ]
)


class EverestRunModelConfig(RunModelConfig):
    experiment_type: Literal["Everest"] = ExperimentType.EVEREST.value
    optimization_output_dir: str
    simulation_dir: str

    parameter_configuration: list[EverestControl]
    response_configuration: list[EverestResponseTypes]

    input_constraints: list[InputConstraintConfig]
    optimization: OptimizationConfig
    model: EverestModelConfig
    keep_run_path: bool
    experiment_name: str
    target_ensemble: str

    def to_experiment_config(self) -> ExperimentConfig:
        return {
            "experiment_type": ExperimentType.EVEREST.value,
            "optimization_output_dir": self.optimization_output_dir,
            "simulation_dir": self.simulation_dir,
            "parameter_configuration": [
                param.model_dump(mode="json") for param in self.parameter_configuration
            ],
            "response_configuration": [
                resp.model_dump(mode="json") for resp in self.response_configuration
            ],
            "input_constraints": [
                c.model_dump(mode="json") for c in self.input_constraints
            ],
            "optimization": self.optimization.model_dump(mode="json"),
            "model": self.model.model_dump(mode="json"),
            "keep_run_path": self.keep_run_path,
            "experiment_name": self.experiment_name,
            "target_ensemble": self.target_ensemble,
        }


class ManualUpdateConfig(UpdateRunModelConfig):
    experiment_type: Literal["Manual Update"] = ExperimentType.MANUAL_UPDATE.value
    ensemble_id: str
    ert_templates: list[tuple[str, str]]

    def to_experiment_config(
        self, *, prior_experiment_config: ExperimentConfig
    ) -> ExperimentConfig:
        return {
            "experiment_type": ExperimentType.MANUAL_UPDATE.value,
            "ensemble_id": self.ensemble_id,
            "ert_templates": self.ert_templates,
            **self._update_experiment_config(),
            "parameter_configuration": prior_experiment_config[
                "parameter_configuration"
            ],
            "response_configuration": prior_experiment_config["response_configuration"],
            "derived_response_configuration": prior_experiment_config.get(
                "derived_response_configuration", []
            ),
            "observations": prior_experiment_config.get("observations", []),
        }


class MultipleDataAssimilationConfig(
    InitialEnsembleRunModelConfig, UpdateRunModelConfig
):
    experiment_type: Literal["Multiple Data Assimilation"] = ExperimentType.ES_MDA.value
    default_weights: ClassVar[str] = "4, 2, 1"
    restart_run: bool
    prior_ensemble_id: str | None
    weights: str

    def to_experiment_config(self) -> ExperimentConfig:
        return {
            "experiment_type": ExperimentType.ES_MDA.value,
            **self._initial_ensemble_experiment_config(),
            **self._update_experiment_config(),
            "restart_run": self.restart_run,
            "prior_ensemble_id": self.prior_ensemble_id,
            "weights": self.weights,
        }


class SingleTestRunConfig(InitialEnsembleRunModelConfig):
    experiment_type: Literal["Single Test Run"] = ExperimentType.SINGLE_TEST_RUN.value
    target_ensemble: str
    supports_rerunning_failed_realizations: ClassVar[bool] = True
    active_realizations: list[bool] = Field(default_factory=lambda: [True])
    minimum_required_realizations: int = 1
