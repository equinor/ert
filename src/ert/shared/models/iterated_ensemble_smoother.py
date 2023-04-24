from __future__ import annotations

import logging
from typing import Any, Dict
from uuid import UUID

from iterative_ensemble_smoother import SIES

from ert._c_wrappers.analysis.analysis_module import AnalysisModule
from ert._c_wrappers.enkf import RunContext
from ert._c_wrappers.enkf.enkf_main import EnKFMain, QueueConfig
from ert._c_wrappers.enkf.enums import HookRuntime, RealizationStateEnum
from ert.analysis import ErtAnalysisError
from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.shared.models import BaseRunModel, ErtRunError
from ert.storage import EnsembleAccessor, EnsembleReader, StorageAccessor

experiment_logger = logging.getLogger("ert.experiment_server.ensemble_experiment")


class IteratedEnsembleSmoother(BaseRunModel):
    def __init__(
        self,
        simulation_arguments: Dict[str, Any],
        ert: EnKFMain,
        storage: StorageAccessor,
        queue_config: QueueConfig,
        experiment_id: UUID,
    ):
        super().__init__(
            simulation_arguments,
            ert,
            storage,
            queue_config,
            experiment_id,
            phase_count=2,
        )
        self.support_restart = False
        analysis_module = ert.resConfig().analysis_config.get_active_module()
        variable_dict = analysis_module.variable_value_dict()
        kwargs = {}
        if "IES_MIN_STEPLENGTH" in variable_dict:
            kwargs["min_steplength"] = variable_dict["IES_MIN_STEPLENGTH"]
        if "IES_MAX_STEPLENGTH" in variable_dict:
            kwargs["max_steplength"] = variable_dict["IES_MAX_STEPLENGTH"]
        if "IES_DEC_STEPLENGTH" in variable_dict:
            kwargs["dec_steplength"] = variable_dict["IES_DEC_STEPLENGTH"]
        self._w_container = SIES(
            len(simulation_arguments["active_realizations"]), **kwargs
        )

    def setAnalysisModule(self, module_name: str) -> AnalysisModule:
        module_load_success = self.ert().analysisConfig().select_module(module_name)

        if not module_load_success:
            raise ErtRunError(f"Unable to load analysis module '{module_name}'!")

        return self.ert().analysisConfig().get_module(module_name)

    def _runAndPostProcess(
        self,
        run_context: RunContext,
        evaluator_server_config: EvaluatorServerConfig,
    ) -> None:
        phase_msg = (
            f"Running iteration {run_context.iteration} of "
            f"{self.phaseCount() - 1} simulation iterations..."
        )
        self.setPhase(run_context.iteration, phase_msg, indeterminate=False)

        self.setPhaseName("Pre processing...", indeterminate=True)
        self.ert().createRunPath(run_context)
        self.ert().runWorkflows(
            HookRuntime.PRE_SIMULATION, self._storage, run_context.sim_fs
        )
        self.setPhaseName("Running forecast...", indeterminate=False)
        num_successful_realizations = self.run_ensemble_evaluator(
            run_context, evaluator_server_config
        )

        self.checkHaveSufficientRealizations(num_successful_realizations)

        self.setPhaseName("Post processing...", indeterminate=True)
        self.ert().runWorkflows(
            HookRuntime.POST_SIMULATION, self._storage, run_context.sim_fs
        )

    def analyzeStep(
        self,
        prior_storage: EnsembleReader,
        posterior_storage: EnsembleAccessor,
        ensemble_id: str,
    ) -> None:
        self.setPhaseName("Analyzing...", indeterminate=True)

        self.setPhaseName("Pre processing update...", indeterminate=True)
        self.ert().runWorkflows(HookRuntime.PRE_UPDATE, self._storage, prior_storage)

        try:
            self.facade.iterative_smoother_update(
                prior_storage, posterior_storage, self._w_container, ensemble_id
            )
        except ErtAnalysisError as e:
            raise ErtRunError(
                f"Analysis of simulation failed with the following error: {e}"
            ) from e

        self.setPhaseName("Post processing update...", indeterminate=True)
        self.ert().runWorkflows(
            HookRuntime.POST_UPDATE, self._storage, posterior_storage
        )

    def runSimulations(
        self, evaluator_server_config: EvaluatorServerConfig
    ) -> RunContext:
        phase_count = self.facade.get_number_of_iterations() + 1
        self.setPhaseCount(phase_count)

        target_case_format = self._simulation_arguments["target_case"]
        prior = self._storage.create_ensemble(
            self._experiment_id,
            name=target_case_format % 0,
            ensemble_size=self._ert.getEnsembleSize(),
        )
        prior_context = self.ert().ensemble_context(
            prior,
            self._simulation_arguments["active_realizations"],
            iteration=0,
        )

        self.ert().analysisConfig().set_case_format(target_case_format)

        self.ert().sample_prior(prior_context.sim_fs, prior_context.active_realizations)
        self._runAndPostProcess(prior_context, evaluator_server_config)

        analysis_config = self.ert().analysisConfig()
        self.ert().runWorkflows(
            HookRuntime.PRE_FIRST_UPDATE, self._storage, prior_context.sim_fs
        )
        for current_iter in range(1, self.facade.get_number_of_iterations() + 1):
            states = [
                RealizationStateEnum.STATE_HAS_DATA,  # type: ignore
                RealizationStateEnum.STATE_INITIALIZED,
            ]
            posterior = self._storage.create_ensemble(
                self._experiment_id,
                name=target_case_format % current_iter,
                ensemble_size=self._ert.getEnsembleSize(),
                iteration=current_iter,
                prior_ensemble=prior_context.sim_fs,
            )
            posterior_context = self.ert().ensemble_context(
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
                self._runAndPostProcess(prior_context, evaluator_server_config)
            if update_success:
                self._runAndPostProcess(posterior_context, evaluator_server_config)
                self.setPhase(phase_count, "Simulations completed.")
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

        return posterior_context

    @classmethod
    def name(cls) -> str:
        return "Iterated ensemble smoother"
