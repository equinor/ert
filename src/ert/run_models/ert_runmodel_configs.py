from collections import defaultdict
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal, Self

import polars as pl
from polars.datatypes import DataTypeClass
from pydantic import BaseModel, Field, field_validator

from ert.base_model_context import BaseModelWithContextSupport
from ert.config import (
    DesignMatrix,
    ESSettings,
    EverestConstraintsConfig,
    EverestObjectivesConfig,
    ExtParamConfig,
    ForwardModelStep,
    GenDataConfig,
    GenKwConfig,
    HookRuntime,
    ModelConfig,
    ObservationSettings,
    QueueConfig,
    SummaryConfig,
    SurfaceConfig,
    Workflow,
)
from ert.config import Field as FieldConfig


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


# https://github.com/pola-rs/polars/issues/13152#issuecomment-1864600078
# PS: Serializing/deserializing schema is scheduled to be added to polars core,
# ref https://github.com/pola-rs/polars/issues/20426
# then this workaround can be omitted.
def str_to_dtype(dtype_str: str) -> pl.DataType:
    dtype = eval(f"pl.{dtype_str}")
    if isinstance(dtype, DataTypeClass):
        dtype = dtype()
    return dtype


class DictEncodedObservations(BaseModel):
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


class InitialEnsembleRunModelConfig(RunModelConfig):
    experiment_name: str
    design_matrix: DesignMatrix | None
    parameter_configuration: list[
        Annotated[
            (GenKwConfig | SurfaceConfig | FieldConfig | ExtParamConfig),
            Field(discriminator="type"),
        ]
    ]
    response_configuration: list[
        Annotated[
            (
                GenDataConfig
                | SummaryConfig
                | EverestConstraintsConfig
                | EverestObjectivesConfig
            ),
            Field(discriminator="type"),
        ]
    ]
    ert_templates: list[tuple[str, str]]
    observations: dict[str, DictEncodedObservations] | None = None

    @field_validator("observations", mode="before")
    def make_dict_encoded_observations(
        cls, v: dict[str, pl.DataFrame | DictEncodedObservations] | None
    ) -> dict[str, DictEncodedObservations] | None:
        if v is None:
            return None
        return {
            k: DictEncodedObservations.from_polars(df)
            if isinstance(df, pl.DataFrame)
            else df
            for k, df in v.items()
        }


class EnsembleExperimentConfig(InitialEnsembleRunModelConfig):
    target_ensemble: str
    supports_rerunning_failed_realizations: ClassVar[bool] = True


class SingleTestRunConfig(EnsembleExperimentConfig):
    active_realizations: list[bool] = Field(default_factory=lambda: [True])
    minimum_required_realizations: int = 1


class UpdateRunModelConfig(RunModelConfig):
    target_ensemble: str
    analysis_settings: ESSettings
    update_settings: ObservationSettings


class EnsembleSmootherConfig(InitialEnsembleRunModelConfig, UpdateRunModelConfig): ...


class EnsembleInformationFilterConfig(
    InitialEnsembleRunModelConfig, UpdateRunModelConfig
): ...


class MultipleDataAssimilationConfig(
    InitialEnsembleRunModelConfig, UpdateRunModelConfig
):
    default_weights: ClassVar[str] = "4, 2, 1"
    restart_run: bool
    prior_ensemble_id: str | None
    weights: str


class EvaluateEnsembleConfig(RunModelConfig):
    ensemble_id: str
    supports_rerunning_failed_realizations: ClassVar[bool] = True


class ManualUpdateConfig(UpdateRunModelConfig):
    ensemble_id: str
