from __future__ import annotations

import functools
import logging
from queue import SimpleQueue
from typing import TYPE_CHECKING
from uuid import UUID

from ert.analysis import ErtAnalysisError, smoother_update
from ert.config import ErtConfig, HookRuntime
from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.run_models.run_arguments import ManualUpdateArguments
from ert.storage import Storage

from ..config.analysis_config import UpdateSettings
from ..config.analysis_module import ESSettings
from .base_run_model import BaseRunModel, ErtRunError, StatusEvents
from .event import RunModelStatusEvent, RunModelUpdateBeginEvent

if TYPE_CHECKING:
    from ert.config import QueueConfig


logger = logging.getLogger(__file__)


class ManualUpdate(BaseRunModel):
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
            config,
            storage,
            queue_config,
            status_queue,
            active_realizations=simulation_arguments.active_realizations,
            total_iterations=1,
            random_seed=simulation_arguments.random_seed,
            minimum_required_realizations=simulation_arguments.minimum_required_realizations,
        )
        self.prior_ensemble_id = simulation_arguments.ensemble_id
        self.target_ensemble_format = simulation_arguments.target_ensemble
        self.ensemble_size = simulation_arguments.ensemble_size
        self.es_settings = es_settings
        self.update_settings = update_settings
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
        self.send_event(RunModelUpdateBeginEvent(iteration=0, run_id=prior.id))

        self._current_iteration_label = "Running update step"
        self.run_workflows(HookRuntime.PRE_FIRST_UPDATE, self._storage, prior)
        self.run_workflows(HookRuntime.PRE_UPDATE, self._storage, prior)

        self.send_event(
            RunModelStatusEvent(
                iteration=0,
                run_id=prior.id,
                msg="Creating posterior ensemble..",
            )
        )
        posterior = self._storage.create_ensemble(
            experiment,
            ensemble_size=prior.ensemble_size,
            iteration=prior.iteration + 1,
            name=ensemble_format % 1,
            prior_ensemble=prior,
        )
        try:
            smoother_update(
                prior,
                posterior,
                analysis_config=self.update_settings,
                es_settings=self.es_settings,
                parameters=prior.experiment.update_parameters,
                observations=prior.experiment.observations.keys(),
                rng=self.rng,
                progress_callback=functools.partial(
                    self.send_smoother_event, 0, prior.id
                ),
            )

        except ErtAnalysisError as e:
            raise ErtRunError(
                f"Analysis of experiment failed with the following error: {e}"
            ) from e

        self.run_workflows(HookRuntime.POST_UPDATE, self._storage, prior)

    @classmethod
    def name(cls) -> str:
        return "Manual update"

    @classmethod
    def description(cls) -> str:
        return "Load parameters and responses from existing → update"
