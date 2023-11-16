from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING
from uuid import UUID

import numpy as np
from iterative_ensemble_smoother import SIES

from ert.analysis import ErtAnalysisError, iterative_smoother_update
from ert.config import ErtConfig, HookRuntime
from ert.enkf_main import sample_prior
from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.realization_state import RealizationState
from ert.run_context import RunContext
from ert.run_models.run_arguments import SIESRunArguments
from ert.storage import EnsembleAccessor, StorageAccessor

from ..analysis._es_update import UpdateSettings
from ..config.analysis_module import IESSettings
from .base_run_model import BaseRunModel, ErtRunError
from .event import (
    RunModelStatusEvent,
    RunModelUpdateBeginEvent,
    RunModelUpdateEndEvent,
)

if TYPE_CHECKING:
    from ert.config import QueueConfig


logger = logging.getLogger(__file__)


class IteratedEnsembleSmoother(BaseRunModel):
    _simulation_arguments: SIESRunArguments

    def __init__(
        self,
        simulation_arguments: SIESRunArguments,
        config: ErtConfig,
        storage: StorageAccessor,
        queue_config: QueueConfig,
        experiment_id: UUID,
        analysis_config: IESSettings,
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
        self.support_restart = False
        self.ies_settings = analysis_config
        self.update_settings = update_settings
        self._w_container = SIES(
            len(simulation_arguments.active_realizations),
            max_steplength=analysis_config.ies_max_steplength,
            min_steplength=analysis_config.ies_min_steplength,
            dec_steplength=analysis_config.ies_dec_steplength,
        )

    def analyzeStep(
        self,
        prior_storage: EnsembleAccessor,
        posterior_storage: EnsembleAccessor,
        ensemble_id: str,
        iteration: int,
    ) -> None:
        self.setPhaseName("Analyzing...", indeterminate=True)

        self.setPhaseName("Pre processing update...", indeterminate=True)
        self.ert.runWorkflows(HookRuntime.PRE_UPDATE, self._storage, prior_storage)
        try:
            iterative_smoother_update(
                prior_storage,
                posterior_storage,
                self._w_container,
                ensemble_id,
                self.ert.update_configuration,
                analysis_config=self.update_settings,
                analysis_settings=self.ies_settings,
                rng=self.rng,
                progress_callback=functools.partial(
                    self.smoother_event_callback, iteration
                ),
                log_path=self.ert_config.analysis_config.log_path,
            )
        except ErtAnalysisError as e:
            raise ErtRunError(
                f"Update algorithm failed with the following error: {e}"
            ) from e

        self.setPhaseName("Post processing update...", indeterminate=True)
        self.ert.runWorkflows(HookRuntime.POST_UPDATE, self._storage, posterior_storage)

    def run_experiment(
        self, evaluator_server_config: EvaluatorServerConfig
    ) -> RunContext:
        self.checkHaveSufficientRealizations(
            self._simulation_arguments.active_realizations.count(True),
            self._simulation_arguments.minimum_required_realizations,
        )
        iteration_count = self.facade.get_number_of_iterations()
        phase_count = iteration_count + 1
        self.setPhaseCount(phase_count)

        log_msg = (
            f"Running SIES for {iteration_count} "
            f'iteration{"s" if (iteration_count != 1) else ""}.'
        )
        logger.info(log_msg)
        self.setPhaseName(log_msg, indeterminate=True)

        target_case_format = self._simulation_arguments.target_case
        prior = self._storage.create_ensemble(
            self._experiment_id,
            ensemble_size=self._simulation_arguments.ensemble_size,
            name=target_case_format % 0,
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

        self.ert.runWorkflows(
            HookRuntime.PRE_FIRST_UPDATE, self._storage, prior_context.sim_fs
        )
        for current_iter in range(1, iteration_count + 1):
            states = [
                RealizationState.HAS_DATA,
                RealizationState.INITIALIZED,
            ]
            self.send_event(RunModelUpdateBeginEvent(iteration=current_iter - 1))
            self.send_event(
                RunModelStatusEvent(
                    iteration=current_iter - 1, msg="Creating posterior ensemble.."
                )
            )

            posterior = self._storage.create_ensemble(
                self._experiment_id,
                name=target_case_format % current_iter,  # noqa
                ensemble_size=prior_context.sim_fs.ensemble_size,
                iteration=current_iter,
                prior_ensemble=prior_context.sim_fs,
            )
            posterior_context = RunContext(
                sim_fs=posterior,
                runpaths=self.run_paths,
                initial_mask=prior_context.sim_fs.get_realization_mask_from_state(
                    states
                ),
                iteration=current_iter,
            )
            update_success = False
            for _iteration in range(self._simulation_arguments.num_retries_per_iter):
                self.analyzeStep(
                    prior_context.sim_fs,
                    posterior_context.sim_fs,
                    str(prior_context.sim_fs.id),
                    current_iter - 1,
                )

                analysis_success = current_iter < self._w_container.iteration_nr
                if analysis_success:
                    update_success = True
                    break
                self._evaluate_and_postprocess(prior_context, evaluator_server_config)
            if update_success:
                self.send_event(RunModelUpdateEndEvent(iteration=current_iter - 1))
                self._evaluate_and_postprocess(
                    posterior_context, evaluator_server_config
                )
            else:
                raise ErtRunError(
                    (
                        "Iterated ensemble smoother stopped: "
                        "maximum number of iteration retries "
                        f"({self._simulation_arguments.num_retries_per_iter} retries) reached "
                        f"for iteration {current_iter}"
                    )
                )
            prior_context = posterior_context

        self.setPhase(phase_count, "Experiment completed.")

        return posterior_context

    @classmethod
    def name(cls) -> str:
        return "Iterated ensemble smoother"
