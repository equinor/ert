from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from uuid import UUID

import numpy as np
from iterative_ensemble_smoother import SIES

from ert.analysis import ErtAnalysisError, ESUpdate
from ert.config import HookRuntime
from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.libres_facade import LibresFacade
from ert.realization_state import RealizationState
from ert.run_context import RunContext
from ert.run_models.run_arguments import SIESRunArguments
from ert.storage import EnsembleAccessor, StorageAccessor

from .base_run_model import BaseRunModel, ErtRunError

if TYPE_CHECKING:
    from ert.config import QueueConfig
    from ert.enkf_main import EnKFMain

logger = logging.getLogger(__file__)


# pylint: disable=too-many-arguments
class IteratedEnsembleSmoother(BaseRunModel):
    _simulation_arguments: SIESRunArguments

    def __init__(
        self,
        simulation_arguments: SIESRunArguments,
        ert: EnKFMain,
        storage: StorageAccessor,
        queue_config: QueueConfig,
        experiment_id: UUID,
    ):
        super().__init__(
            simulation_arguments,
            ert,
            LibresFacade(ert),
            storage,
            queue_config,
            experiment_id,
            phase_count=2,
        )
        self.support_restart = False
        analysis_module = ert.resConfig().analysis_config.active_module()
        variable_dict = analysis_module.variable_value_dict()
        kwargs = {}
        if "IES_MIN_STEPLENGTH" in variable_dict:
            kwargs["min_steplength"] = variable_dict["IES_MIN_STEPLENGTH"]
        if "IES_MAX_STEPLENGTH" in variable_dict:
            kwargs["max_steplength"] = variable_dict["IES_MAX_STEPLENGTH"]
        if "IES_DEC_STEPLENGTH" in variable_dict:
            kwargs["dec_steplength"] = variable_dict["IES_DEC_STEPLENGTH"]
        self._w_container = SIES(
            len(simulation_arguments.active_realizations), **kwargs
        )

    def analyzeStep(
        self,
        prior_storage: EnsembleAccessor,
        posterior_storage: EnsembleAccessor,
        ensemble_id: str,
    ) -> None:
        self.setPhaseName("Analyzing...", indeterminate=True)

        self.setPhaseName("Pre processing update...", indeterminate=True)
        self.ert.runWorkflows(HookRuntime.PRE_UPDATE, self._storage, prior_storage)

        smoother = ESUpdate(self.ert)
        try:
            smoother.iterative_smoother_update(
                prior_storage, posterior_storage, self._w_container, ensemble_id
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
            self._simulation_arguments.active_realizations.count(True)
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
            ensemble_size=self.ert.getEnsembleSize(),
            name=target_case_format % 0,
        )
        self.set_env_key("_ERT_ENSEMBLE_ID", str(prior.id))
        prior_context = self.ert.ensemble_context(
            prior,
            np.array(self._simulation_arguments.active_realizations, dtype=bool),
            iteration=0,
        )

        self.ert.analysisConfig().set_case_format(target_case_format)

        self.ert.sample_prior(prior_context.sim_fs, prior_context.active_realizations)
        self._evaluate_and_postprocess(prior_context, evaluator_server_config)

        analysis_config = self.ert.analysisConfig()
        self.ert.runWorkflows(
            HookRuntime.PRE_FIRST_UPDATE, self._storage, prior_context.sim_fs
        )
        for current_iter in range(1, iteration_count + 1):
            states = [
                RealizationState.HAS_DATA,
                RealizationState.INITIALIZED,
            ]
            posterior = self._storage.create_ensemble(
                self._experiment_id,
                name=target_case_format % current_iter,  # noqa
                ensemble_size=self.ert.getEnsembleSize(),
                iteration=current_iter,
                prior_ensemble=prior_context.sim_fs,
            )
            posterior_context = self.ert.ensemble_context(
                posterior,
                prior_context.sim_fs.get_realization_mask_from_state(states),
                iteration=current_iter,
            )
            update_success = False
            for _iteration in range(analysis_config.num_retries_per_iter):
                self.analyzeStep(
                    prior_context.sim_fs,
                    posterior_context.sim_fs,
                    str(prior_context.sim_fs.id),
                )

                analysis_success = current_iter < self._w_container.iteration_nr
                if analysis_success:
                    update_success = True
                    break
                self._evaluate_and_postprocess(prior_context, evaluator_server_config)
            if update_success:
                self._evaluate_and_postprocess(
                    posterior_context, evaluator_server_config
                )
            else:
                raise ErtRunError(
                    (
                        "Iterated ensemble smoother stopped: "
                        "maximum number of iteration retries "
                        f"({analysis_config.num_retries_per_iter} retries) reached "
                        f"for iteration {current_iter}"
                    )
                )
            prior_context = posterior_context
        self.setPhase(phase_count, "Experiment completed.")

        return posterior_context

    @classmethod
    def name(cls) -> str:
        return "Iterated ensemble smoother"
