from typing import Any, Dict, Optional

from ert.analysis import ErtAnalysisError
from ert_shared.ensemble_evaluator.config import EvaluatorServerConfig
from ert_shared.models import BaseRunModel, ErtRunError
from res.enkf import EnkfSimulationRunner, RunContext
from res.enkf.enkf_main import EnKFMain, QueueConfig
from res.enkf.enums import HookRuntime, RealizationStateEnum


class EnsembleSmoother(BaseRunModel):
    def __init__(
        self,
        simulation_arguments: Dict[str, Any],
        ert: EnKFMain,
        queue_config: QueueConfig,
    ):
        super().__init__(simulation_arguments, ert, queue_config, phase_count=2)
        self.support_restart = False

    def setAnalysisModule(self, module_name: str) -> None:
        module_load_success = self.ert().analysisConfig().selectModule(module_name)

        if not module_load_success:
            raise ErtRunError(f"Unable to load analysis module '{module_name}'!")

    def runSimulations(
        self, evaluator_server_config: EvaluatorServerConfig
    ) -> RunContext:
        prior_context = self.create_context()

        self._checkMinimumActiveRealizations(prior_context)
        self.setPhase(0, "Running simulations...", indeterminate=False)

        # self.setAnalysisModule(arguments["analysis_module"])

        self.setPhaseName("Pre processing...", indeterminate=True)
        self.ert().getEnkfSimulationRunner().createRunPath(prior_context)

        EnkfSimulationRunner.runWorkflows(HookRuntime.PRE_SIMULATION, ert=self.ert())

        # Push ensemble, parameters, observations to new storage
        ensemble_id = self._post_ensemble_data()

        self.setPhaseName("Running forecast...", indeterminate=False)

        num_successful_realizations = self.run_ensemble_evaluator(
            prior_context, evaluator_server_config
        )

        # Push simulation results to storage
        self._post_ensemble_results(ensemble_id)

        self.checkHaveSufficientRealizations(num_successful_realizations)

        self.setPhaseName("Post processing...", indeterminate=True)
        EnkfSimulationRunner.runWorkflows(HookRuntime.POST_SIMULATION, ert=self.ert())

        self.setPhaseName("Analyzing...")
        EnkfSimulationRunner.runWorkflows(HookRuntime.PRE_FIRST_UPDATE, ert=self.ert())
        EnkfSimulationRunner.runWorkflows(HookRuntime.PRE_UPDATE, ert=self.ert())
        try:
            self.facade.smoother_update(prior_context)
        except ErtAnalysisError as e:
            raise ErtRunError(
                f"Analysis of simulation failed with the following error: {e}"
            ) from e

        EnkfSimulationRunner.runWorkflows(HookRuntime.POST_UPDATE, ert=self.ert())

        # Create an update object in storage
        analysis_module_name = self.ert().analysisConfig().activeModuleName()
        update_id = self._post_update_data(ensemble_id, analysis_module_name)

        self.setPhase(1, "Running simulations...")
        self.ert().getEnkfFsManager().switchFileSystem(prior_context.target_fs)

        self.setPhaseName("Pre processing...")

        rerun_context = self.create_context(prior_context=prior_context)

        self.ert().getEnkfSimulationRunner().createRunPath(rerun_context)

        EnkfSimulationRunner.runWorkflows(HookRuntime.PRE_SIMULATION, ert=self.ert())
        # Push ensemble, parameters, observations to new storage
        ensemble_id = self._post_ensemble_data(update_id=update_id)

        self.setPhaseName("Running forecast...", indeterminate=False)

        num_successful_realizations = self.run_ensemble_evaluator(
            rerun_context, evaluator_server_config
        )

        self.checkHaveSufficientRealizations(num_successful_realizations)

        self.setPhaseName("Post processing...", indeterminate=True)
        EnkfSimulationRunner.runWorkflows(HookRuntime.POST_SIMULATION, ert=self.ert())

        # Push simulation results to storage
        self._post_ensemble_results(ensemble_id)

        self.setPhase(2, "Simulations completed.")

        return prior_context

    def create_context(self, prior_context: Optional[RunContext] = None) -> RunContext:

        fs_manager = self.ert().getEnkfFsManager()
        if prior_context is None:
            sim_fs = fs_manager.getCurrentFileSystem()
            target_fs = fs_manager.getFileSystem(
                self._simulation_arguments["target_case"]
            )
            itr = 0
            mask = self._simulation_arguments["active_realizations"]
        else:
            itr = 1
            sim_fs = prior_context.target_fs
            target_fs = None
            mask = sim_fs.getStateMap().createMask(
                RealizationStateEnum.STATE_HAS_DATA
                | RealizationStateEnum.STATE_INITIALIZED
            )

        run_context = self.ert().create_ensemble_smoother_run_context(
            active_mask=mask,
            iteration=itr,
            target_filesystem=target_fs,
            source_filesystem=sim_fs,
        )
        self._run_context = run_context
        self._last_run_iteration = run_context.iteration
        return run_context

    @classmethod
    def name(cls) -> str:
        return "Ensemble smoother"
