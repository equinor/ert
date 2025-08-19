from typing import Annotated

from pydantic import Field

from ert.config import (
    EverestConstraintsConfig,
    EverestObjectivesConfig,
    ExtParamConfig,
    GenDataConfig,
    SummaryConfig,
)
from everest.config import (
    ControlConfig,
    InputConstraintConfig,
    ObjectiveFunctionConfig,
    OptimizationConfig,
    OutputConstraintConfig,
)
from everest.config import ModelConfig as ModelConfigEverest

from .ert_runmodel_configs import RunModelConfig


class EverestRunModelConfig(RunModelConfig):
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

    experiment_name: str
    target_ensemble: str
