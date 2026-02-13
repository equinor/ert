from __future__ import annotations

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


class UpdateRunModelConfig(RunModelConfig):
    target_ensemble: str
    analysis_settings: ESSettings
    update_settings: ObservationSettings


class EnsembleSmootherConfig(InitialEnsembleRunModelConfig, UpdateRunModelConfig):
    type: Literal["ensemble_smoother"] = "ensemble_smoother"


class EnsembleInformationFilterConfig(
    InitialEnsembleRunModelConfig, UpdateRunModelConfig
):
    type: Literal["ensemble_information_filter"] = "ensemble_information_filter"


class EnsembleExperimentConfig(InitialEnsembleRunModelConfig):
    type: Literal["ensemble_experiment"] = "ensemble_experiment"
    target_ensemble: str
    supports_rerunning_failed_realizations: ClassVar[bool] = True


class EvaluateEnsembleConfig(RunModelConfig):
    type: Literal["evaluate_ensemble"] = "evaluate_ensemble"
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
    type: Literal["everest"] = "everest"
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


class ManualUpdateConfig(UpdateRunModelConfig):
    type: Literal["manual_update"] = "manual_update"
    ensemble_id: str
    ert_templates: list[tuple[str, str]]


class MultipleDataAssimilationConfig(
    InitialEnsembleRunModelConfig, UpdateRunModelConfig
):
    type: Literal["es_mda"] = "es_mda"
    default_weights: ClassVar[str] = "4, 2, 1"
    restart_run: bool
    prior_ensemble_id: str | None
    weights: str


class SingleTestRunConfig(InitialEnsembleRunModelConfig):
    type: Literal["single_test_run"] = "single_test_run"
    target_ensemble: str
    supports_rerunning_failed_realizations: ClassVar[bool] = True
    active_realizations: list[bool] = Field(default_factory=lambda: [True])
    minimum_required_realizations: int = 1
