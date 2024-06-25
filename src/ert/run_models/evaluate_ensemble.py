from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

import numpy as np

from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.run_context import RunContext
from ert.run_models.run_arguments import EvaluateEnsembleRunArguments
from ert.storage import Ensemble, Storage

from . import BaseRunModel

if TYPE_CHECKING:
    from queue import SimpleQueue

    from ert.config import ErtConfig, QueueConfig

    from .base_run_model import StatusEvents


# pylint: disable=too-many-arguments
class EvaluateEnsemble(BaseRunModel):
    """
    This workflow will evaluate ensembles which have parameters, but no
    simulation has been performed, so there are no responses. This can
    be used in instances where the parameters are sampled manually, or
    after performing a manual update step. This will always read parameter
    and response configuration from the stored ensemble, and will not
    reflect any changes to the user configuration on disk.
    """

    def __init__(
        self,
        simulation_arguments: EvaluateEnsembleRunArguments,
        config: ErtConfig,
        storage: Storage,
        queue_config: QueueConfig,
        status_queue: SimpleQueue[StatusEvents],
    ):
        super().__init__(
            config,
            storage,
            queue_config,
            status_queue,
            active_realizations=simulation_arguments.active_realizations,
            minimum_required_realizations=simulation_arguments.minimum_required_realizations,
        )
        self.ensemble_id = simulation_arguments.ensemble_id

    def run_experiment(
        self, evaluator_server_config: EvaluatorServerConfig, restart: bool = False
    ) -> RunContext:
        self.setPhaseName("Running evaluate experiment...")

        ensemble_id = self.ensemble_id
        ensemble_uuid = UUID(ensemble_id)
        ensemble = self._storage.get_ensemble(ensemble_uuid)
        assert isinstance(ensemble, Ensemble)

        experiment = ensemble.experiment
        self.set_env_key("_ERT_EXPERIMENT_ID", str(experiment.id))
        self.set_env_key("_ERT_ENSEMBLE_ID", str(ensemble.id))

        prior_context = RunContext(
            ensemble=ensemble,
            runpaths=self.run_paths,
            initial_mask=np.array(self.active_realizations, dtype=bool),
            iteration=ensemble.iteration,
        )

        iteration = prior_context.iteration
        phase_count = iteration + 1
        self.setPhaseCount(phase_count)
        self._evaluate_and_postprocess(prior_context, evaluator_server_config)

        self.setPhase(phase_count, "Simulations completed.")

        return prior_context

    @classmethod
    def name(cls) -> str:
        return "Evaluate ensemble"
