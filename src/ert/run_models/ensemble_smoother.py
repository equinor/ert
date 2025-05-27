from __future__ import annotations

import logging
from pathlib import Path
from queue import SimpleQueue
from typing import TYPE_CHECKING

from ert.config import ErtConfig, ESSettings, ObservationSettings
from ert.storage import Storage

from .base_run_model import StatusEvents
from .update_run_model import UpdateRunModel

if TYPE_CHECKING:
    from ert.config import QueueConfig


logger = logging.getLogger(__name__)


class EnsembleSmoother(UpdateRunModel):
    def __init__(
        self,
        target_ensemble: str,
        experiment_name: str,
        active_realizations: list[bool],
        minimum_required_realizations: int,
        random_seed: int,
        config: ErtConfig,
        storage: Storage,
        queue_config: QueueConfig,
        es_settings: ESSettings,
        update_settings: ObservationSettings,
        status_queue: SimpleQueue[StatusEvents],
    ):
        super().__init__(
            es_settings,
            update_settings,
            storage,
            config.runpath_file,
            Path(config.user_config_file),
            config.env_vars,
            config.env_pr_fm_step,
            config.runpath_config,
            queue_config,
            config.forward_model_steps,
            status_queue,
            config.substitutions,
            config.hooked_workflows,
            active_realizations=active_realizations,
            start_iteration=0,
            total_iterations=2,
            random_seed=random_seed,
            minimum_required_realizations=minimum_required_realizations,
            log_path=config.analysis_config.log_path,
            config=config,
            target_ensemble=target_ensemble,
            experiment_name=experiment_name,
        )

    @classmethod
    def name(cls) -> str:
        return "Ensemble smoother"

    @classmethod
    def description(cls) -> str:
        return "Sample parameters → evaluate → update → evaluate"
