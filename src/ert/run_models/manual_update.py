from __future__ import annotations

import logging
from queue import SimpleQueue
from typing import TYPE_CHECKING, List, Optional
from uuid import UUID

from ert.config import ErtConfig
from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.storage import Storage

from ..config.analysis_config import UpdateSettings
from ..config.analysis_module import ESSettings
from .base_run_model import ErtRunError, StatusEvents, UpdateRunModel

if TYPE_CHECKING:
    from ert.config import QueueConfig


logger = logging.getLogger(__file__)


class ManualUpdate(UpdateRunModel):
    """Manual update"""

    def __init__(
        self,
        ensemble_id: str,
        target_ensemble: str,
        active_realizations: List[bool],
        minimum_required_realizations: int,
        random_seed: Optional[int],
        config: ErtConfig,
        storage: Storage,
        queue_config: QueueConfig,
        es_settings: ESSettings,
        update_settings: UpdateSettings,
        status_queue: SimpleQueue[StatusEvents],
    ):
        super().__init__(
            es_settings,
            update_settings,
            config,
            storage,
            queue_config,
            status_queue,
            active_realizations=active_realizations,
            total_iterations=1,
            start_iteration=0,
            random_seed=random_seed,
            minimum_required_realizations=minimum_required_realizations,
        )
        self.prior_ensemble_id = ensemble_id
        self.target_ensemble_format = target_ensemble
        self.support_restart = False

    def run_experiment(
        self, evaluator_server_config: EvaluatorServerConfig, restart: bool = False
    ) -> None:
        logger.info("Running manual update")
        ensemble_format = self.target_ensemble_format
        try:
            ensemble_id = UUID(self.prior_ensemble_id)
            prior = self._storage.get_ensemble(ensemble_id)
            experiment = prior.experiment
            self.set_env_key("_ERT_EXPERIMENT_ID", str(experiment.id))
            self.set_env_key("_ERT_ENSEMBLE_ID", str(prior.id))
        except (KeyError, ValueError) as err:
            raise ErtRunError(
                f"Prior ensemble with ID: {ensemble_id} does not exists"
            ) from err
        self.update(prior, ensemble_format % (prior.iteration + 1))

    @classmethod
    def name(cls) -> str:
        return "Manual update"

    @classmethod
    def description(cls) -> str:
        return "Load parameters and responses from existing → update"
