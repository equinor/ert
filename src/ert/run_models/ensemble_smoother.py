from __future__ import annotations

import logging
from queue import SimpleQueue
from typing import TYPE_CHECKING, List, Optional

import numpy as np

from ert.config import ErtConfig
from ert.enkf_main import sample_prior
from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.storage import Storage

from ..config.analysis_config import UpdateSettings
from ..config.analysis_module import ESSettings
from ..run_arg import create_run_arguments
from .base_run_model import StatusEvents, UpdateRunModel

if TYPE_CHECKING:
    from ert.config import QueueConfig


logger = logging.getLogger(__name__)


class EnsembleSmoother(UpdateRunModel):
    def __init__(
        self,
        target_ensemble: str,
        experiment_name: str,
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
            start_iteration=0,
            total_iterations=2,
            random_seed=random_seed,
            minimum_required_realizations=minimum_required_realizations,
        )
        self.target_ensemble_format = target_ensemble
        self.experiment_name = experiment_name

        self.support_restart = False

    def run_experiment(
        self, evaluator_server_config: EvaluatorServerConfig, restart: bool = False
    ) -> None:
        self.log_at_startup()
        ensemble_format = self.target_ensemble_format
        experiment = self._storage.create_experiment(
            parameters=self.ert_config.ensemble_config.parameter_configuration,
            observations=self.ert_config.observations,
            responses=self.ert_config.ensemble_config.response_configuration,
            name=self.experiment_name,
        )

        self.set_env_key("_ERT_EXPERIMENT_ID", str(experiment.id))
        prior = self._storage.create_ensemble(
            experiment,
            ensemble_size=self.ensemble_size,
            name=ensemble_format % 0,
        )
        self.set_env_key("_ERT_ENSEMBLE_ID", str(prior.id))
        prior_args = create_run_arguments(
            self.run_paths,
            np.array(self.active_realizations, dtype=bool),
            ensemble=prior,
        )

        sample_prior(
            prior,
            np.where(self.active_realizations)[0],
            random_seed=self.random_seed,
        )

        self._evaluate_and_postprocess(
            prior_args,
            prior,
            evaluator_server_config,
        )
        posterior = self.update(prior, ensemble_format % 1)

        posterior_args = create_run_arguments(
            self.run_paths,
            np.array(self.active_realizations, dtype=bool),
            ensemble=posterior,
        )

        self._evaluate_and_postprocess(
            posterior_args,
            posterior,
            evaluator_server_config,
        )

    @classmethod
    def name(cls) -> str:
        return "Ensemble smoother"

    @classmethod
    def description(cls) -> str:
        return "Sample parameters → evaluate → update → evaluate"
