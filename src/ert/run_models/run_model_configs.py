from __future__ import annotations

import dataclasses
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal, Self

import polars as pl
from polars.datatypes import DataTypeClass
from pydantic import BaseModel, Field, TypeAdapter, model_validator

from ert.base_model_context import BaseModelWithContextSupport, init_context_var
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
from ert.config.forward_model_step import (
    SiteOrUserForwardModelStep,
    UserInstalledForwardModelStep,
)
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
    forward_model_steps: list[SiteOrUserForwardModelStep]
    substitutions: dict[str, str]
    hooked_workflows: defaultdict[HookRuntime, list[Workflow]]
    active_realizations: list[bool]
    log_path: Path
    random_seed: int
    start_iteration: int = 0
    minimum_required_realizations: int = 0
    supports_rerunning_failed_realizations: ClassVar[bool] = False

    @model_validator(mode="after")
    def _restore_plugin_forward_model_step_subclasses(self) -> Self:
        runtime_plugins = init_context_var.get()
        if runtime_plugins is None:
            return self
        restored = []
        for step in self.forward_model_steps:
            installed = runtime_plugins.installed_forward_model_steps.get(step.name)
            if installed is not None and not isinstance(
                step, (UserInstalledForwardModelStep, type(installed))
            ):
                fm_step = installed.model_copy(
                    update={
                        field: getattr(step, field)
                        for field in ForwardModelStep.model_fields
                    }
                )
                restored.append(fm_step)
            else:
                restored.append(step)
        self.forward_model_steps = restored
        return self


class InitialEnsembleRunModelConfig(RunModelConfig):
    experiment_name: str
    design_matrix: DictEncodedDataFrame | None = None
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
            some_json = self.design_matrix.model_dump(mode="json")
            print(f"{some_json=}")
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
    experiment_type: ExperimentType = ExperimentType.ENSEMBLE_SMOOTHER

    def to_experiment_config(self) -> ExperimentConfig:
        return {
            "experiment_type": ExperimentType.ENSEMBLE_SMOOTHER,
            **self._initial_ensemble_experiment_config(),
            **self._update_experiment_config(),
        }


class EnsembleInformationFilterConfig(
    InitialEnsembleRunModelConfig, UpdateRunModelConfig
):
    experiment_type: ExperimentType = ExperimentType.ENSEMBLE_INFORMATION_FILTER

    def to_experiment_config(self) -> ExperimentConfig:
        return {
            "experiment_type": ExperimentType.ENSEMBLE_INFORMATION_FILTER,
            **self._initial_ensemble_experiment_config(),
            **self._update_experiment_config(),
        }


class EnsembleExperimentConfig(InitialEnsembleRunModelConfig):
    experiment_type: ExperimentType = ExperimentType.ENSEMBLE_EXPERIMENT
    target_ensemble: str
    supports_rerunning_failed_realizations: ClassVar[bool] = True

    def to_experiment_config(self) -> ExperimentConfig:
        return {
            "experiment_type": self.experiment_type,
            **self._initial_ensemble_experiment_config(),
            "target_ensemble": self.target_ensemble,
        }


class EvaluateEnsembleConfig(RunModelConfig):
    experiment_type: ExperimentType = ExperimentType.EVALUATE_ENSEMBLE
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
    experiment_type: ExperimentType = ExperimentType.EVEREST
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
            "experiment_type": ExperimentType.EVEREST,
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
    experiment_type: ExperimentType = ExperimentType.MANUAL_UPDATE
    ensemble_id: str
    ert_templates: list[tuple[str, str]]

    def to_experiment_config(
        self, *, prior_experiment_config: ExperimentConfig
    ) -> ExperimentConfig:
        return {
            "experiment_type": ExperimentType.MANUAL_UPDATE,
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
    experiment_type: ExperimentType = ExperimentType.ES_MDA
    default_weights: ClassVar[str] = "4, 2, 1"
    restart_run: bool
    prior_ensemble_id: str | None
    weights: str

    def to_experiment_config(self) -> ExperimentConfig:
        return {
            "experiment_type": ExperimentType.ES_MDA,
            **self._initial_ensemble_experiment_config(),
            **self._update_experiment_config(),
            "restart_run": self.restart_run,
            "prior_ensemble_id": self.prior_ensemble_id,
            "weights": self.weights,
        }


class SingleTestRunConfig(InitialEnsembleRunModelConfig):
    experiment_type: ExperimentType = ExperimentType.SINGLE_TEST_RUN
    target_ensemble: str
    supports_rerunning_failed_realizations: ClassVar[bool] = True
    active_realizations: list[bool] = Field(default_factory=lambda: [True])
    minimum_required_realizations: int = 1
