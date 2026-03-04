from __future__ import annotations

import functools
import logging

import numpy as np
from pydantic import PrivateAttr

from ert.config import (
    PostExperimentFixtures,
    PreExperimentFixtures,
)
from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.run_models.initial_ensemble_run_model import (
    InitialEnsembleRunModel,
    InitialEnsembleRunModelConfig,
)
from ert.run_models.update_run_model import UpdateRunModel, UpdateRunModelConfig
from ert.storage import Ensemble
from ert.trace import tracer

from ..analysis import build_strategy_map, smoother_update
from ..run_arg import create_run_arguments
from .run_model import ErtRunError, ExperimentType

logger = logging.getLogger(__name__)


class EnsembleSmootherConfig(InitialEnsembleRunModelConfig, UpdateRunModelConfig): ...


class EnsembleSmoother(InitialEnsembleRunModel, UpdateRunModel, EnsembleSmootherConfig):
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
            experiment_config=self.model_dump(mode="json"),
            name=self.experiment_name,
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
        progress_callback = functools.partial(
            self.send_smoother_event,
            prior.iteration,
            prior.id,
        )
        strategy_map = build_strategy_map(
            parameters=prior.experiment.update_parameters,
            param_configs=prior.experiment.parameter_configuration,
            inversion=self.analysis_settings.inversion,
            enkf_truncation=self.analysis_settings.enkf_truncation,
            distance_localization=self.analysis_settings.distance_localization,
            localization=self.analysis_settings.localization,
            correlation_threshold=self.analysis_settings.correlation_threshold,
            rng=self._rng,
            progress_callback=progress_callback,
        )
        smoother_update(
            prior,
            posterior,
            update_settings=self.update_settings,
            strategy_map=strategy_map,
            parameters=prior.experiment.update_parameters,
            observations=prior.experiment.observation_keys,
            global_scaling=weight,
            progress_callback=progress_callback,
            active_realizations=self.active_realizations,
        )

    @classmethod
    def name(cls) -> str:
        return "Ensemble smoother"

    @classmethod
    def description(cls) -> str:
        return "Sample parameters → evaluate → update → evaluate"

    @classmethod
    def _experiment_type(cls) -> ExperimentType:
        return ExperimentType.ENSEMBLE_SMOOTHER
