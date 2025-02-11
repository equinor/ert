from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

import numpy as np

from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.storage import Storage
from ert.trace import tracer

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
    This workflow will evaluate ensembles which have parameters, but no simulation has been performed, so there are no responses.<br>
    This can be used in instances where the parameters are sampled manually, or after performing a manual update step.<br>
    The workflow will always read parameter and response configuration from the stored ensemble,<br>
    and will not reflect any changes to the user configuration on disk.<br>
    """

    def __init__(
        self,
        active_realizations: list[bool],
        minimum_required_realizations: int,
        ensemble_id: str,
        random_seed: int | None,
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
            start_iteration=self.ensemble.iteration,
            total_iterations=1,
            active_realizations=active_realizations,
            minimum_required_realizations=minimum_required_realizations,
            random_seed=random_seed,
            log_path=config.analysis_config.log_path,
        )

    @tracer.start_as_current_span(f"{__name__}.run_experiment")
    def run_experiment(
        self, evaluator_server_config: EvaluatorServerConfig, restart: bool = False
    ) -> None:
        self.log_at_startup()
        self.restart = restart
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
