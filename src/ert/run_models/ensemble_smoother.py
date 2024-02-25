from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING

import numpy as np

from ert.analysis import ErtAnalysisError, smoother_update
from ert.config import ErtConfig, HookRuntime
from ert.enkf_main import sample_prior
from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.run_context import RunContext
from ert.run_models.run_arguments import ESRunArguments
from ert.storage import Storage

from ..config.analysis_config import UpdateSettings
from ..config.analysis_module import ESSettings
from .base_run_model import BaseRunModel, ErtRunError
from .event import RunModelStatusEvent, RunModelUpdateBeginEvent, RunModelUpdateEndEvent

if TYPE_CHECKING:
    from ert.config import QueueConfig


logger = logging.getLogger(__file__)


class EnsembleSmoother(BaseRunModel):
    def __init__(
        self,
        simulation_arguments: ESRunArguments,
        config: ErtConfig,
        storage: Storage,
        queue_config: QueueConfig,
        es_settings: ESSettings,
        update_settings: UpdateSettings,
    ):
        super().__init__(
            simulation_arguments,
            config,
            storage,
            queue_config,
            phase_count=2,
        )
        self.es_settings = es_settings
        self.update_settings = update_settings
        self.support_restart = False

    @property
    def simulation_arguments(self) -> ESRunArguments:
        args = self._simulation_arguments
        assert isinstance(args, ESRunArguments)
        return args

    def run_experiment(
        self, evaluator_server_config: EvaluatorServerConfig
    ) -> RunContext:
        self.checkHaveSufficientRealizations(
            self._simulation_arguments.active_realizations.count(True),
            self._simulation_arguments.minimum_required_realizations,
        )

        log_msg = "Running ES"
        logger.info(log_msg)
        self.setPhaseName(log_msg, indeterminate=True)
        experiment = self._storage.create_experiment(
            parameters=self.ert_config.ensemble_config.parameter_configuration,
            observations=self.ert_config.observations,
            responses=self.ert_config.ensemble_config.response_configuration,
            simulation_arguments=self._simulation_arguments,
            name=self._simulation_arguments.experiment_name,
        )

        self.set_env_key("_ERT_EXPERIMENT_ID", str(experiment.id))
        prior_name = self._simulation_arguments.current_ensemble
        prior = self._storage.create_ensemble(
            experiment,
            ensemble_size=self._simulation_arguments.ensemble_size,
            name=prior_name,
        )
        self.set_env_key("_ERT_ENSEMBLE_ID", str(prior.id))
        prior_context = RunContext(
            ensemble=prior,
            runpaths=self.run_paths,
            initial_mask=np.array(
                self._simulation_arguments.active_realizations, dtype=bool
            ),
            iteration=0,
        )

        sample_prior(
            prior_context.ensemble,
            prior_context.active_realizations,
            random_seed=self._simulation_arguments.random_seed,
        )

        self._evaluate_and_postprocess(prior_context, evaluator_server_config)

        self.send_event(RunModelUpdateBeginEvent(iteration=0))

        self.setPhaseName("Running ES update step")
        self.ert.runWorkflows(
            HookRuntime.PRE_FIRST_UPDATE, self._storage, prior_context.ensemble
        )
        self.ert.runWorkflows(
            HookRuntime.PRE_UPDATE, self._storage, prior_context.ensemble
        )

        self.send_event(
            RunModelStatusEvent(iteration=0, msg="Creating posterior ensemble..")
        )
        target_ensemble_format = self._simulation_arguments.target_ensemble
        posterior_context = RunContext(
            ensemble=self._storage.create_ensemble(
                experiment,
                ensemble_size=prior.ensemble_size,
                iteration=1,
                name=target_ensemble_format,
                prior_ensemble=prior,
            ),
            runpaths=self.run_paths,
            initial_mask=(
                prior_context.ensemble.get_realization_mask_with_parameters()
                * prior_context.ensemble.get_realization_mask_with_responses()
                * prior_context.ensemble.get_realization_mask_without_failure()
            ),
            iteration=1,
        )
        try:
            smoother_snapshot = smoother_update(
                prior_context.ensemble,
                posterior_context.ensemble,
                prior_context.run_id,  # type: ignore
                analysis_config=self.update_settings,
                es_settings=self.es_settings,
                parameters=prior_context.ensemble.experiment.update_parameters,
                observations=prior_context.ensemble.experiment.observations.keys(),
                rng=self.rng,
                progress_callback=functools.partial(self.smoother_event_callback, 0),
                log_path=self.ert_config.analysis_config.log_path,
            )

        except ErtAnalysisError as e:
            raise ErtRunError(
                f"Analysis of experiment failed with the following error: {e}"
            ) from e

        self.send_event(
            RunModelUpdateEndEvent(iteration=0, smoother_snapshot=smoother_snapshot)
        )

        self.ert.runWorkflows(
            HookRuntime.POST_UPDATE, self._storage, posterior_context.ensemble
        )

        self._evaluate_and_postprocess(posterior_context, evaluator_server_config)

        self.setPhase(2, "Experiment completed.")

        return posterior_context

    @classmethod
    def name(cls) -> str:
        return "Ensemble smoother"
