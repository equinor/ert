from __future__ import annotations

from queue import SimpleQueue
from typing import TYPE_CHECKING

import numpy as np

from ert.enkf_main import sample_prior
from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.run_context import RunContext
from ert.storage import Ensemble, Experiment, Storage

from .base_run_model import BaseRunModel, StatusEvents

if TYPE_CHECKING:
    from ert.config import ErtConfig, QueueConfig
    from ert.run_models.run_arguments import (
        EnsembleExperimentRunArguments,
    )


class EnsembleExperiment(BaseRunModel):
    """
    This workflow will create a new experiment and a new ensemble from
    the user configuration. It will never overwrite existing ensembles, and
    will always sample parameters.
    """

    def __init__(
        self,
        simulation_arguments: EnsembleExperimentRunArguments,
        config: ErtConfig,
        storage: Storage,
        queue_config: QueueConfig,
        status_queue: SimpleQueue[StatusEvents],
    ):
        self.ensemble_name = simulation_arguments.ensemble_name
        self.experiment_name = simulation_arguments.experiment_name
        self.ensemble_size = simulation_arguments.ensemble_size
        self.experiment: Experiment | None = None
        self.ensemble: Ensemble | None = None

        super().__init__(
            config,
            storage,
            queue_config,
            status_queue,
            active_realizations=simulation_arguments.active_realizations,
            random_seed=simulation_arguments.random_seed,
            minimum_required_realizations=simulation_arguments.minimum_required_realizations,
        )

    def run_experiment(
        self,
        evaluator_server_config: EvaluatorServerConfig,
        restart: bool = False,
    ) -> RunContext:
        self.setPhaseName(self.run_message())
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

        run_context = RunContext(
            ensemble=self.ensemble,
            runpaths=self.run_paths,
            initial_mask=np.array(self.active_realizations, dtype=bool),
        )
        sample_prior(
            run_context.ensemble,
            run_context.active_realizations,
            random_seed=self.random_seed,
        )

        phase_count = run_context.iteration + 1
        self.setPhaseCount(phase_count)

        self._evaluate_and_postprocess(run_context, evaluator_server_config)

        self.setPhase(phase_count, "Simulations completed.")

        return run_context

    @classmethod
    def run_message(cls) -> str:
        return "Running ensemble experiment..."

    @classmethod
    def name(cls) -> str:
        return "Ensemble experiment"
