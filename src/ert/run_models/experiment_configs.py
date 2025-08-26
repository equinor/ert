from collections import defaultdict
from pathlib import Path
from typing import Annotated, Any, ClassVar

from pydantic import BaseModel, Field

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
from ert.storage.local_experiment import DictEncodedObservations
from everest.config import (
    ControlConfig,
    InputConstraintConfig,
    ObjectiveFunctionConfig,
    OptimizationConfig,
    OutputConstraintConfig,
)
from everest.config import ModelConfig as ModelConfigEverest


class ExperimentConfig(BaseModel):
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


class ExperimentWithInitialEnsembleConfig(ExperimentConfig):
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


class EnsembleExperimentConfig(ExperimentWithInitialEnsembleConfig):
    target_ensemble: str
    supports_rerunning_failed_realizations: ClassVar[bool] = True


class SingleTestRunConfig(EnsembleExperimentConfig):
    active_realizations: list[bool] = Field(default_factory=lambda: [True])
    minimum_required_realizations: int = 1


class ExperimentWithUpdateConfig(ExperimentConfig):
    target_ensemble: str
    analysis_settings: ESSettings
    update_settings: ObservationSettings


class EnsembleSmootherConfig(
    ExperimentWithInitialEnsembleConfig, ExperimentWithUpdateConfig
): ...


class EnsembleInformationFilterConfig(
    ExperimentWithInitialEnsembleConfig, ExperimentWithUpdateConfig
): ...


class MultipleDataAssimilationConfig(
    ExperimentWithInitialEnsembleConfig, ExperimentWithUpdateConfig
):
    default_weights: ClassVar[str] = "4, 2, 1"
    restart_run: bool
    prior_ensemble_id: str | None
    weights: str


class EvaluateEnsembleConfig(ExperimentConfig):
    ensemble_id: str
    supports_rerunning_failed_realizations: ClassVar[bool] = True


class ManualUpdateConfig(ExperimentWithUpdateConfig):
    ensemble_id: str


class EverestExperimentConfig(ExperimentConfig):
    optimization_output_dir: str
    simulation_dir: str

    parameter_configuration: list[ExtParamConfig]
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

    controls: list[ControlConfig]

    objective_functions: list[ObjectiveFunctionConfig]
    objective_names: list[str]

    input_constraints: list[InputConstraintConfig]

    output_constraints: list[OutputConstraintConfig]
    constraint_names: list[str]

    optimization: OptimizationConfig

    model: ModelConfigEverest
    keep_run_path: bool
