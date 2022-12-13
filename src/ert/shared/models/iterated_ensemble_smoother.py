import logging
from typing import Any, Dict, Optional

from iterative_ensemble_smoother import IterativeEnsembleSmoother

from ert._c_wrappers.analysis.analysis_module import AnalysisModule
from ert._c_wrappers.enkf import RunContext
from ert._c_wrappers.enkf.enkf_fs import EnkfFs
from ert._c_wrappers.enkf.enkf_main import EnKFMain, QueueConfig
from ert._c_wrappers.enkf.enums import HookRuntime, RealizationStateEnum
from ert.analysis import ErtAnalysisError
from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.shared.models import BaseRunModel, ErtRunError

experiment_logger = logging.getLogger("ert.experiment_server.ensemble_experiment")


class IteratedEnsembleSmoother(BaseRunModel):
    def __init__(
        self,
        simulation_arguments: Dict[str, Any],
        ert: EnKFMain,
        queue_config: QueueConfig,
        id_: str,
    ):
        super().__init__(simulation_arguments, ert, queue_config, id_, phase_count=2)
        self.support_restart = False
        self._w_container = IterativeEnsembleSmoother(
            len(simulation_arguments["active_realizations"])
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
        update_id: Optional[str] = None,
    ) -> str:
        phase_msg = (
            f"Running iteration {run_context.iteration} of "
            f"{self.phaseCount() - 1} simulation iterations..."
        )
        self.setPhase(run_context.iteration, phase_msg, indeterminate=False)

        self.setPhaseName("Pre processing...", indeterminate=True)
        self.ert().createRunPath(run_context)
        self.ert().runWorkflows(HookRuntime.PRE_SIMULATION)
        # create ensemble
        ensemble_id = self._post_ensemble_data(
            case_name=run_context.sim_fs.case_name, update_id=update_id
        )
        self.setPhaseName("Running forecast...", indeterminate=False)
        num_successful_realizations = self.run_ensemble_evaluator(
            run_context, evaluator_server_config
        )

        self.checkHaveSufficientRealizations(num_successful_realizations)

        self.setPhaseName("Post processing...", indeterminate=True)
        self.ert().runWorkflows(HookRuntime.POST_SIMULATION)
        self._post_ensemble_results(run_context.sim_fs.case_name, ensemble_id)
        return ensemble_id

    def analyzeStep(
        self, prior_storage: "EnkfFs", posterior_storage: "EnkfFs", ensemble_id: str
    ) -> str:
        self.setPhaseName("Analyzing...", indeterminate=True)

        self.setPhaseName("Pre processing update...", indeterminate=True)
        self.ert().runWorkflows(HookRuntime.PRE_UPDATE)

        try:
            self.facade.iterative_smoother_update(
                prior_storage, posterior_storage, self._w_container, ensemble_id
            )
        except ErtAnalysisError as e:
            raise ErtRunError(
                f"Analysis of simulation failed with the following error: {e}"
            ) from e

        # Push update data to new storage
        analysis_module_name = self.ert().analysisConfig().active_module_name()
        update_id = self._post_update_data(
            parent_ensemble_id=ensemble_id, algorithm=analysis_module_name
        )

        self.setPhaseName("Post processing update...", indeterminate=True)
        self.ert().runWorkflows(HookRuntime.POST_UPDATE)
        return update_id

    def runSimulations(
        self, evaluator_server_config: EvaluatorServerConfig
    ) -> RunContext:
        phase_count = self.facade.get_number_of_iterations() + 1
        self.setPhaseCount(phase_count)

        target_case_format = self._simulation_arguments["target_case"]
        prior_context = self.ert().create_ensemble_context(
            target_case_format % 0,
            self._simulation_arguments["active_realizations"],
            iteration=0,
        )

        self.ert().analysisConfig().set_case_format(target_case_format)

        self.ert().sample_prior(prior_context.sim_fs, prior_context.active_realizations)
        ensemble_id = self._runAndPostProcess(prior_context, evaluator_server_config)

        analysis_config = self.ert().analysisConfig()
        self.ert().runWorkflows(HookRuntime.PRE_FIRST_UPDATE)
        for current_iter in range(1, self.facade.get_number_of_iterations() + 1):
            state = (
                RealizationStateEnum.STATE_HAS_DATA  # type: ignore
                | RealizationStateEnum.STATE_INITIALIZED
            )
            posterior_context = self.ert().create_ensemble_context(
                target_case_format % current_iter,
                prior_context.sim_fs.getStateMap().createMask(state),
                iteration=current_iter,
            )
            update_success = False
            for iteration in range(analysis_config.num_retries_per_iter):
                update_id = self.analyzeStep(
                    prior_context.sim_fs,
                    posterior_context.sim_fs,
                    ensemble_id,
                )

                analysis_success = current_iter < self._w_container.iteration_nr
                if analysis_success:
                    update_success = True
                    break
                ensemble_id = self._runAndPostProcess(
                    prior_context, evaluator_server_config, update_id
                )
            if update_success:
                ensemble_id = self._runAndPostProcess(
                    posterior_context, evaluator_server_config
                )
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
