from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional
from uuid import UUID

import numpy as np

from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.storage import Storage

from ..run_arg import create_run_arguments
from . import BaseRunModel

if TYPE_CHECKING:
    from queue import SimpleQueue

    from ert.config import ErtConfig, QueueConfig

    from .base_run_model import StatusEvents

logger = logging.getLogger(__name__)


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
        active_realizations: List[bool],
        minimum_required_realizations: int,
        ensemble_id: str,
        random_seed: Optional[int],
        config: ErtConfig,
        storage: Storage,
        queue_config: QueueConfig,
        status_queue: SimpleQueue[StatusEvents],
    ):
        try:
            self.ensemble = storage.get_ensemble(UUID(ensemble_id))
        except KeyError as err:
            raise ValueError(f"No ensemble: {ensemble_id}") from err
        super().__init__(
            config,
            storage,
            queue_config,
            status_queue,
            start_iteration=self.ensemble.iteration,
            total_iterations=1,
            active_realizations=active_realizations,
            minimum_required_realizations=minimum_required_realizations,
            random_seed=random_seed,
        )

    def run_experiment(
        self, evaluator_server_config: EvaluatorServerConfig, restart: bool = False
    ) -> None:
        self.log_at_startup()
        ensemble = self.ensemble
        experiment = ensemble.experiment
        self.set_env_key("_ERT_EXPERIMENT_ID", str(experiment.id))
        self.set_env_key("_ERT_ENSEMBLE_ID", str(ensemble.id))

        prior_args = create_run_arguments(
            self.run_paths,
            np.array(self.active_realizations, dtype=bool),
            ensemble=ensemble,
        )

        self._total_iterations = ensemble.iteration + 1
        self._evaluate_and_postprocess(
            prior_args,
            ensemble,
            evaluator_server_config,
        )

    @classmethod
    def name(cls) -> str:
        return "Evaluate ensemble"

    @classmethod
    def description(cls) -> str:
        return "Take existing ensemble parameters → evaluate"
