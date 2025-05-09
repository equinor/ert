from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

import numpy as np

from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.trace import tracer

from ..run_arg import create_run_arguments
from ..storage import open_storage
from . import BaseRunModel
from .experiment_configs import RunModelConfig

if TYPE_CHECKING:
    from queue import SimpleQueue

    from ert.config import ErtConfig, QueueConfig

    from .base_run_model import StatusEvents

logger = logging.getLogger(__name__)


class EvaluateEnsemble(BaseRunModel):
    """
    This workflow will evaluate ensembles which have parameters, but no simulation
    has been performed, so there are no responses. This can be used in instances
    where the parameters are sampled manually, or after performing a manual update step.
    The workflow will always read parameter and response configuration from the stored
    ensemble, and will not reflect any changes to the user configuration on disk.
    """

    def __init__(
        self,
        active_realizations: list[bool],
        minimum_required_realizations: int,
        ensemble_id: str,
        random_seed: int,
        config: ErtConfig,
        storage_path: str,
        queue_config: QueueConfig,
        status_queue: SimpleQueue[StatusEvents],
    ):
        try:
            with open_storage(storage_path, mode="r") as storage:
                self.ensemble = storage.get_ensemble(UUID(ensemble_id))
        except KeyError as err:
            raise ValueError(f"No ensemble: {ensemble_id}") from err

        super().__init__(
            RunModelConfig(
                storage_path=storage_path,
                runpath_file=config.runpath_file,
                user_config_file=Path(config.user_config_file),
                env_vars=config.env_vars,
                env_pr_fm_step=config.env_pr_fm_step,
                runpath_config=config.runpath_config,
                queue_config=queue_config,
                forward_model_steps=config.forward_model_steps,
                substitutions=config.substitutions,
                hooked_workflows=config.hooked_workflows,
                start_iteration=self.ensemble.iteration,
                total_iterations=1,
                active_realizations=active_realizations,
                minimum_required_realizations=minimum_required_realizations,
                random_seed=random_seed,
                log_path=config.analysis_config.log_path,
            ),
            status_queue=status_queue,
        )

    @tracer.start_as_current_span(f"{__name__}.run_experiment")
    def run_experiment(
        self, evaluator_server_config: EvaluatorServerConfig, restart: bool = False
    ) -> None:
        self.log_at_startup()
        self.restart = restart
        if self.restart:
            self.active_realizations = self._create_mask_from_failed_realizations()
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
        return "Use existing parameters → evaluate"
