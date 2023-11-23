from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING
from uuid import UUID

import numpy as np

from ert.analysis import ErtAnalysisError, smoother_update
from ert.config import ErtConfig, HookRuntime
from ert.enkf_main import sample_prior
from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.realization_state import RealizationState
from ert.run_context import RunContext
from ert.run_models.run_arguments import ESRunArguments
from ert.storage import StorageAccessor

from ..analysis._es_update import UpdateSettings
from ..config.analysis_module import ESSettings
from .base_run_model import BaseRunModel, ErtRunError
from .event import (
    RunModelStatusEvent,
    RunModelUpdateBeginEvent,
    RunModelUpdateEndEvent,
)

if TYPE_CHECKING:
    from ert.config import QueueConfig


logger = logging.getLogger(__file__)


class EnsembleSmoother(BaseRunModel):
    def __init__(
        self,
        simulation_arguments: ESRunArguments,
        config: ErtConfig,
        storage: StorageAccessor,
        queue_config: QueueConfig,
        experiment_id: UUID,
        es_settings: ESSettings,
        update_settings: UpdateSettings,
    ):
        super().__init__(
            simulation_arguments,
            config,
            storage,
            queue_config,
            experiment_id,
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

        prior_name = self._simulation_arguments.current_case
        prior = self._storage.create_ensemble(
            self._experiment_id,
            ensemble_size=self._simulation_arguments.ensemble_size,
            name=prior_name,
        )
        self.set_env_key("_ERT_ENSEMBLE_ID", str(prior.id))
        prior_context = RunContext(
            sim_fs=prior,
            runpaths=self.run_paths,
            initial_mask=np.array(
                self._simulation_arguments.active_realizations, dtype=bool
            ),
            iteration=0,
        )

        sample_prior(
            prior_context.sim_fs,
            prior_context.active_realizations,
            random_seed=self._simulation_arguments.random_seed,
        )

        self._evaluate_and_postprocess(prior_context, evaluator_server_config)

        self.send_event(RunModelUpdateBeginEvent(iteration=0))

        self.setPhaseName("Running ES update step")
        self.ert.runWorkflows(
            HookRuntime.PRE_FIRST_UPDATE, self._storage, prior_context.sim_fs
        )
        self.ert.runWorkflows(
            HookRuntime.PRE_UPDATE, self._storage, prior_context.sim_fs
        )

        states = [
            RealizationState.HAS_DATA,
            RealizationState.INITIALIZED,
        ]

        self.send_event(
            RunModelStatusEvent(iteration=0, msg="Creating posterior ensemble..")
        )
        target_case_format = self._simulation_arguments.target_case
        posterior_context = RunContext(
            sim_fs=self._storage.create_ensemble(
                self._experiment_id,
                ensemble_size=prior.ensemble_size,
                iteration=1,
                name=target_case_format,
                prior_ensemble=prior,
            ),
            runpaths=self.run_paths,
            initial_mask=prior_context.sim_fs.get_realization_mask_from_state(states),
            iteration=1,
        )
        try:
            smoother_update(
                prior_context.sim_fs,
                posterior_context.sim_fs,
                prior_context.run_id,  # type: ignore
                analysis_config=self.update_settings,
                es_settings=self.es_settings,
                updatestep=self.ert.update_configuration,
                rng=self.rng,
                progress_callback=functools.partial(self.smoother_event_callback, 0),
                log_path=self.ert_config.analysis_config.log_path,
            )
        except ErtAnalysisError as e:
            raise ErtRunError(
                f"Analysis of experiment failed with the following error: {e}"
            ) from e

        self.send_event(RunModelUpdateEndEvent(iteration=0))

        self.ert.runWorkflows(
            HookRuntime.POST_UPDATE, self._storage, posterior_context.sim_fs
        )

        self._evaluate_and_postprocess(posterior_context, evaluator_server_config)

        self.setPhase(2, "Experiment completed.")

        return posterior_context

    @classmethod
    def name(cls) -> str:
        return "Ensemble smoother"
