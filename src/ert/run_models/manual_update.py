from __future__ import annotations

import logging
from queue import SimpleQueue
from typing import TYPE_CHECKING
from uuid import UUID

from ert.config import ErtConfig
from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.run_models.run_arguments import ManualUpdateArguments
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
        simulation_arguments: ManualUpdateArguments,
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
            active_realizations=simulation_arguments.active_realizations,
            total_iterations=1,
            start_iteration=0,
            random_seed=simulation_arguments.random_seed,
            minimum_required_realizations=simulation_arguments.minimum_required_realizations,
        )
        self.prior_ensemble_id = simulation_arguments.ensemble_id
        self.target_ensemble_format = simulation_arguments.target_ensemble
        self.ensemble_size = simulation_arguments.ensemble_size
        self.support_restart = False

    def run_experiment(
        self, evaluator_server_config: EvaluatorServerConfig, restart: bool = False
    ) -> None:
        log_msg = "Running manual update"
        logger.info(log_msg)
        self._current_iteration_label = log_msg
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
