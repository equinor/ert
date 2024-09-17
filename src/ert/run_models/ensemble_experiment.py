from __future__ import annotations

import logging
from queue import SimpleQueue
from typing import TYPE_CHECKING, List, Optional

import numpy as np

from ert.enkf_main import sample_prior
from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.storage import Ensemble, Experiment, Storage

from ..run_arg import create_run_arguments
from .base_run_model import BaseRunModel, StatusEvents

if TYPE_CHECKING:
    from ert.config import ErtConfig, QueueConfig


logger = logging.getLogger(__name__)


class EnsembleExperiment(BaseRunModel):
    """
    This workflow will create a new experiment and a new ensemble from
    the user configuration. It will never overwrite existing ensembles, and
    will always sample parameters.
    """

    def __init__(
        self,
        ensemble_name: str,
        experiment_name: str,
        active_realizations: List[bool],
        minimum_required_realizations: int,
        random_seed: Optional[int],
        config: ErtConfig,
        storage: Storage,
        queue_config: QueueConfig,
        status_queue: SimpleQueue[StatusEvents],
    ):
        self.ensemble_name = ensemble_name
        self.experiment_name = experiment_name
        self.experiment: Experiment | None = None
        self.ensemble: Ensemble | None = None

        super().__init__(
            config,
            storage,
            queue_config,
            status_queue,
            total_iterations=1,
            active_realizations=active_realizations,
            random_seed=random_seed,
            minimum_required_realizations=minimum_required_realizations,
        )

    def run_experiment(
        self,
        evaluator_server_config: EvaluatorServerConfig,
        restart: bool = False,
    ) -> None:
        self.log_at_startup()
        if not restart:
            self.experiment = self._storage.create_experiment(
                name=self.experiment_name,
                parameters=self.ert_config.ensemble_config.parameter_configuration,
                observations=self.ert_config.observations,
                responses=self.ert_config.ensemble_config.response_configuration,
            )
            self.ensemble = self._storage.create_ensemble(
                self.experiment,
                name=self.ensemble_name,
                ensemble_size=self.ensemble_size,
            )
        else:
            self.active_realizations = self._create_mask_from_failed_realizations()

        assert self.experiment
        assert self.ensemble

        self.set_env_key("_ERT_EXPERIMENT_ID", str(self.experiment.id))
        self.set_env_key("_ERT_ENSEMBLE_ID", str(self.ensemble.id))

        run_args = create_run_arguments(
            self.run_paths,
            np.array(self.active_realizations, dtype=bool),
            ensemble=self.ensemble,
        )
        sample_prior(
            self.ensemble,
            np.where(self.active_realizations)[0],
            random_seed=self.random_seed,
        )

        self._evaluate_and_postprocess(
            run_args,
            self.ensemble,
            evaluator_server_config,
        )

    @classmethod
    def name(cls) -> str:
        return "Ensemble experiment"

    @classmethod
    def description(cls) -> str:
        return "Sample parameters → evaluate (N realizations)"
