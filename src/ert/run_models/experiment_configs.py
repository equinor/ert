from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ert.config import (
    ESSettings,
    ForwardModelStep,
    HookRuntime,
    ModelConfig,
    ObservationSettings,
    QueueConfig,
    Workflow,
)
from ert.substitutions import Substitutions


class RunModelConfig(BaseModel):
    storage_path: str
    runpath_file: Path
    user_config_file: Path
    env_vars: dict[str, str]
    env_pr_fm_step: dict[str, dict[str, Any]]
    runpath_config: ModelConfig
    queue_config: QueueConfig
    forward_model_steps: list[ForwardModelStep]
    substitutions: Substitutions
    hooked_workflows: defaultdict[HookRuntime, list[Workflow]]
    active_realizations: list[bool]
    log_path: Path
    random_seed: int
    total_iterations: int = 1
    start_iteration: int = 0
    minimum_required_realizations: int = 0


class UpdateRunModelConfig(RunModelConfig):
    analysis_settings: ESSettings
    update_settings: ObservationSettings

    def to_run_model_config(self) -> RunModelConfig:
        base_fields = RunModelConfig.model_fields.keys()
        base_kwargs = {field: getattr(self, field) for field in base_fields}
        return RunModelConfig(**base_kwargs)
