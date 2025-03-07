from __future__ import annotations

import logging
from pathlib import Path
from queue import SimpleQueue
from typing import TYPE_CHECKING
from uuid import UUID

from ert.config import ErtConfig, ESSettings, UpdateSettings
from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.storage import Storage

from .base_run_model import ErtRunError, StatusEvents, UpdateRunModel

if TYPE_CHECKING:
    from ert.config import QueueConfig


logger = logging.getLogger(__name__)


class ManualUpdate(UpdateRunModel):
    """Manual update"""

    def __init__(
        self,
        ensemble_id: str,
        target_ensemble: str,
        active_realizations: list[bool],
        minimum_required_realizations: int,
        random_seed: int | None,
        config: ErtConfig,
        storage: Storage,
        queue_config: QueueConfig,
        es_settings: ESSettings,
        update_settings: UpdateSettings,
        status_queue: SimpleQueue[StatusEvents],
    ):
        try:
            prior = storage.get_ensemble(UUID(ensemble_id))
        except (KeyError, ValueError) as err:
            raise ErtRunError(
                f"Prior ensemble with ID: {UUID(ensemble_id)} does not exists"
            ) from err

        super().__init__(
            es_settings,
            update_settings,
            storage,
            config.runpath_file,
            Path(config.user_config_file),
            config.env_vars,
            config.env_pr_fm_step,
            config.model_config,
            queue_config,
            config.forward_model_steps,
            status_queue,
            config.substitutions,
            config.ert_templates,
            config.hooked_workflows,
            active_realizations=active_realizations,
            total_iterations=1,
            start_iteration=prior.iteration,
            random_seed=random_seed,
            minimum_required_realizations=minimum_required_realizations,
            log_path=config.analysis_config.log_path,
        )
        self.prior = prior
        self.target_ensemble_format = target_ensemble
        self.support_restart = False

    def run_experiment(
        self, evaluator_server_config: EvaluatorServerConfig, restart: bool = False
    ) -> None:
        self.log_at_startup()
        self.set_env_key("_ERT_EXPERIMENT_ID", str(self.prior.experiment.id))
        self.set_env_key("_ERT_ENSEMBLE_ID", str(self.prior.id))

        ensemble_format = self.target_ensemble_format
        self.update(self.prior, ensemble_format % (self.prior.iteration + 1))

    @classmethod
    def name(cls) -> str:
        return "Manual update"

    @classmethod
    def description(cls) -> str:
        return "Load parameters and responses from existing → update"

    def check_if_runpath_exists(self) -> bool:
        # Will not run a forward model, so does not create files on runpath
        return False
