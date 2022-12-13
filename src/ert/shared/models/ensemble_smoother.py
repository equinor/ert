import logging
from typing import Any, Dict

from ert._c_wrappers.enkf import RunContext
from ert._c_wrappers.enkf.enkf_main import EnKFMain, QueueConfig
from ert._c_wrappers.enkf.enums import HookRuntime, RealizationStateEnum
from ert.analysis import ErtAnalysisError
from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.shared.models import BaseRunModel, ErtRunError

experiment_logger = logging.getLogger("ert.experiment_server.ensemble_experiment")


class EnsembleSmoother(BaseRunModel):
    def __init__(
        self,
        simulation_arguments: Dict[str, Any],
        ert: EnKFMain,
        queue_config: QueueConfig,
        id_: str,
    ):
        super().__init__(simulation_arguments, ert, queue_config, id_, phase_count=2)
        self.support_restart = False

    def setAnalysisModule(self, module_name: str) -> None:
        module_load_success = self.ert().analysisConfig().select_module(module_name)

        if not module_load_success:
            raise ErtRunError(f"Unable to load analysis module '{module_name}'!")

    def runSimulations(
        self, evaluator_server_config: EvaluatorServerConfig
    ) -> RunContext:
        prior_name = self._simulation_arguments.get(
            "current_case", self.ert().storage_manager.active_case
        )
        if prior_name not in self.ert().storage_manager:
            self.ert().storage_manager.add_case(prior_name)
        prior_context = self.ert().load_ensemble_context(
            prior_name,
            self._simulation_arguments["active_realizations"],
            iteration=0,
        )

        self._checkMinimumActiveRealizations(len(prior_context.active_realizations))
        self.setPhase(0, "Running simulations...", indeterminate=False)

        # self.setAnalysisModule(arguments["analysis_module"])

        self.setPhaseName("Pre processing...", indeterminate=True)
        self.ert().sample_prior(prior_context.sim_fs, prior_context.active_realizations)
        self.ert().createRunPath(prior_context)

        self.ert().runWorkflows(HookRuntime.PRE_SIMULATION)

        # Push ensemble, parameters, observations to new storage
        ensemble_id = self._post_ensemble_data(prior_context.sim_fs.case_name)

        self.setPhaseName("Running forecast...", indeterminate=False)

        num_successful_realizations = self.run_ensemble_evaluator(
            prior_context, evaluator_server_config
        )

        # Push simulation results to storage
        self._post_ensemble_results(prior_context.sim_fs.case_name, ensemble_id)

        self.checkHaveSufficientRealizations(num_successful_realizations)

        self.setPhaseName("Post processing...", indeterminate=True)
        self.ert().runWorkflows(HookRuntime.POST_SIMULATION)

        self.setPhaseName("Analyzing...")
        self.ert().runWorkflows(HookRuntime.PRE_FIRST_UPDATE)
        self.ert().runWorkflows(HookRuntime.PRE_UPDATE)
        state = (
            RealizationStateEnum.STATE_HAS_DATA  # type: ignore
            | RealizationStateEnum.STATE_INITIALIZED
        )
        target_case_format = self._simulation_arguments["target_case"]
        posterior_context = self.ert().create_ensemble_context(
            target_case_format,
            prior_context.sim_fs.getStateMap().createMask(state),
            iteration=1,
        )

        try:
            self.facade.smoother_update(
                prior_context.sim_fs,
                posterior_context.sim_fs,
                prior_context.run_id,
            )
        except ErtAnalysisError as e:
            raise ErtRunError(
                f"Analysis of simulation failed with the following error: {e}"
            ) from e

        self.ert().runWorkflows(HookRuntime.POST_UPDATE)

        # Create an update object in storage
        analysis_module_name = self.ert().analysisConfig().active_module_name()
        update_id = self._post_update_data(ensemble_id, analysis_module_name)

        self.setPhase(1, "Running simulations...")

        self.setPhaseName("Pre processing...")

        self.ert().createRunPath(posterior_context)

        self.ert().runWorkflows(HookRuntime.PRE_SIMULATION)
        # Push ensemble, parameters, observations to new storage
        ensemble_id = self._post_ensemble_data(
            case_name=posterior_context.sim_fs.case_name, update_id=update_id
        )

        self.setPhaseName("Running forecast...", indeterminate=False)

        num_successful_realizations = self.run_ensemble_evaluator(
            posterior_context, evaluator_server_config
        )

        self.checkHaveSufficientRealizations(num_successful_realizations)

        self.setPhaseName("Post processing...", indeterminate=True)
        self.ert().runWorkflows(HookRuntime.POST_SIMULATION)

        # Push simulation results to storage
        self._post_ensemble_results(posterior_context.sim_fs.case_name, ensemble_id)

        self.setPhase(2, "Simulations completed.")

        return posterior_context

    @classmethod
    def name(cls) -> str:
        return "Ensemble smoother"
