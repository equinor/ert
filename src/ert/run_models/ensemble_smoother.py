from __future__ import annotations

import functools
import logging
from typing import cast

import numpy as np
from pydantic import PrivateAttr

from ert.config import (
    ParameterConfig,
    PostExperimentFixtures,
    PreExperimentFixtures,
    ResponseConfig,
)
from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.run_models.initial_ensemble_run_model import InitialEnsembleRunModel
from ert.run_models.update_run_model import UpdateRunModel
from ert.storage import Ensemble
from ert.trace import tracer

from ..analysis import smoother_update
from ..run_arg import create_run_arguments
from .run_model import ErtRunError

logger = logging.getLogger(__name__)


class EnsembleSmoother(UpdateRunModel, InitialEnsembleRunModel):
    _total_iterations: int = PrivateAttr(default=2)

    @tracer.start_as_current_span(f"{__name__}.run_experiment")
    def run_experiment(
        self,
        evaluator_server_config: EvaluatorServerConfig,
        rerun_failed_realizations: bool = False,
    ) -> None:
        self.log_at_startup()
        if rerun_failed_realizations:
            raise ErtRunError("Ensemble Information Filter does not support restart")

        self.run_workflows(fixtures=PreExperimentFixtures(random_seed=self.random_seed))

        experiment_storage = self._storage.create_experiment(
            parameters=cast(list[ParameterConfig], self.parameter_configuration),
            observations={k: v.to_polars() for k, v in self.observations.items()}
            if self.observations is not None
            else None,
            responses=cast(list[ResponseConfig], self.response_configuration),
            name=self.experiment_name,
            templates=self.ert_templates,
        )

        prior = self._storage.create_ensemble(
            experiment_storage,
            ensemble_size=self.ensemble_size,
            name=self.target_ensemble % 0,
        )

        self.set_env_key("_ERT_EXPERIMENT_ID", str(prior.experiment.id))
        self.set_env_key("_ERT_ENSEMBLE_ID", str(prior.id))

        self._sample_and_evaluate_ensemble(evaluator_server_config, prior)

        posterior = self.update(prior, self.target_ensemble % 1)

        posterior_args = create_run_arguments(
            self._run_paths,
            np.array(self.active_realizations, dtype=bool),
            ensemble=posterior,
        )

        self._evaluate_and_postprocess(
            posterior_args,
            posterior,
            evaluator_server_config,
        )
        self.run_workflows(
            fixtures=PostExperimentFixtures(
                random_seed=self.random_seed,
                storage=self._storage,
                ensemble=posterior,
            ),
        )

    def update_ensemble_parameters(
        self, prior: Ensemble, posterior: Ensemble, weight: float
    ) -> None:
        smoother_update(
            prior,
            posterior,
            update_settings=self.update_settings,
            es_settings=self.analysis_settings,
            parameters=prior.experiment.update_parameters,
            observations=prior.experiment.observation_keys,
            global_scaling=weight,
            rng=self._rng,
            progress_callback=functools.partial(
                self.send_smoother_event,
                prior.iteration,
                prior.id,
            ),
            active_realizations=self.active_realizations,
        )

    @classmethod
    def name(cls) -> str:
        return "Ensemble smoother"

    @classmethod
    def description(cls) -> str:
        return "Sample parameters → evaluate → update → evaluate"
