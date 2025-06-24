from __future__ import annotations

import logging
from typing import Any, ClassVar
from uuid import UUID

import numpy as np

from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.trace import tracer

from ..run_arg import create_run_arguments
from .run_model import RunModel

logger = logging.getLogger(__name__)


class EvaluateEnsemble(RunModel):
    """
    This workflow will evaluate ensembles which have parameters, but no simulation
    has been performed, so there are no responses. This can be used in instances
    where the parameters are sampled manually, or after performing a manual update step.
    The workflow will always read parameter and response configuration from the stored
    ensemble, and will not reflect any changes to the user configuration on disk.
    """

    ensemble_id: str
    supports_rerunning_failed_realizations: ClassVar[bool] = True

    def model_post_init(self, ctx: Any) -> None:
        super().model_post_init(ctx)
        try:
            ensemble = self._storage.get_ensemble(UUID(self.ensemble_id))
            self.start_iteration = ensemble.iteration
        except KeyError as err:
            raise ValueError(f"No ensemble: {self.ensemble_id}") from err

    @tracer.start_as_current_span(f"{__name__}.run_experiment")
    def run_experiment(
        self,
        evaluator_server_config: EvaluatorServerConfig,
        rerun_failed_realizations: bool = False,
    ) -> None:
        self.log_at_startup()
        self._is_rerunning_failed_realizations = rerun_failed_realizations
        ensemble = self._storage.get_ensemble(UUID(self.ensemble_id))
        self.set_env_key("_ERT_EXPERIMENT_ID", str(ensemble.experiment.id))
        self.set_env_key("_ERT_ENSEMBLE_ID", str(ensemble.id))

        prior_args = create_run_arguments(
            self._run_paths,
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
