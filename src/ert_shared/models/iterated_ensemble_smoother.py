from typing import Any, Dict, Optional

from ert.analysis import ErtAnalysisError, ModuleData
from ert_shared.ensemble_evaluator.config import EvaluatorServerConfig
from ert_shared.models import BaseRunModel, ErtRunError
from res.analysis.analysis_module import AnalysisModule
from res.enkf import EnkfSimulationRunner, RunContext
from res.enkf.enkf_fs import EnkfFs
from res.enkf.enkf_main import EnKFMain, QueueConfig
from res.enkf.enums import HookRuntime, RealizationStateEnum


class IteratedEnsembleSmoother(BaseRunModel):
    def __init__(
        self,
        simulation_arguments: Dict[str, Any],
        ert: EnKFMain,
        queue_config: QueueConfig,
    ):
        super().__init__(simulation_arguments, ert, queue_config, phase_count=2)
        self.support_restart = False
        self._w_container = ModuleData(len(simulation_arguments["active_realizations"]))

    def setAnalysisModule(self, module_name: str) -> AnalysisModule:
        module_load_success = self.ert().analysisConfig().selectModule(module_name)

        if not module_load_success:
            raise ErtRunError(f"Unable to load analysis module '{module_name}'!")

        return self.ert().analysisConfig().getModule(module_name)

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
        self.ert().getEnkfSimulationRunner().createRunPath(run_context)
        EnkfSimulationRunner.runWorkflows(HookRuntime.PRE_SIMULATION, ert=self.ert())
        # create ensemble
        ensemble_id = self._post_ensemble_data(update_id=update_id)
        self.setPhaseName("Running forecast...", indeterminate=False)
        num_successful_realizations = self.run_ensemble_evaluator(
            run_context, evaluator_server_config
        )

        self.checkHaveSufficientRealizations(num_successful_realizations)

        self.setPhaseName("Post processing...", indeterminate=True)
        EnkfSimulationRunner.runWorkflows(HookRuntime.POST_SIMULATION, ert=self.ert())
        self._post_ensemble_results(ensemble_id)
        return ensemble_id

    def createTargetCaseFileSystem(self, phase: int, target_case_format: str) -> EnkfFs:
        target_fs = (
            self.ert().getEnkfFsManager().getFileSystem(target_case_format % phase)
        )
        return target_fs

    def analyzeStep(self, run_context: RunContext, ensemble_id: str) -> str:
        self.setPhaseName("Analyzing...", indeterminate=True)

        self.setPhaseName("Pre processing update...", indeterminate=True)
        EnkfSimulationRunner.runWorkflows(HookRuntime.PRE_UPDATE, ert=self.ert())

        try:
            self.facade.iterative_smoother_update(run_context, self._w_container)
            self._w_container.iteration_nr += 1
        except ErtAnalysisError as e:
            raise ErtRunError(
                f"Analysis of simulation failed with the following error: {e}"
            ) from e

        # Push update data to new storage
        analysis_module_name = self.ert().analysisConfig().activeModuleName()
        update_id = self._post_update_data(
            parent_ensemble_id=ensemble_id, algorithm=analysis_module_name
        )

        self.setPhaseName("Post processing update...", indeterminate=True)
        EnkfSimulationRunner.runWorkflows(HookRuntime.POST_UPDATE, ert=self.ert())
        return update_id

    def runSimulations(
        self, evaluator_server_config: EvaluatorServerConfig
    ) -> RunContext:
        phase_count = self.facade.get_number_of_iterations() + 1
        self.setPhaseCount(phase_count)

        target_case_format = self._simulation_arguments["target_case"]
        run_context = self.create_context(0)

        self.ert().analysisConfig().getAnalysisIterConfig().setCaseFormat(
            target_case_format
        )

        ensemble_id = self._runAndPostProcess(run_context, evaluator_server_config)

        analysis_config = self.ert().analysisConfig()
        analysis_iter_config = analysis_config.getAnalysisIterConfig()
        num_retries_per_iteration = analysis_iter_config.getNumRetries()
        num_retries = 0
        current_iter = 0

        while (
            current_iter < self.facade.get_number_of_iterations()
            and num_retries < num_retries_per_iteration
        ):
            pre_analysis_iter_num = self._w_container.iteration_nr - 1
            # We run the PRE_FIRST_UPDATE hook here because the current_iter is
            # explicitly available, versus in the run_context inside analyzeStep
            if current_iter == 0:
                EnkfSimulationRunner.runWorkflows(
                    HookRuntime.PRE_FIRST_UPDATE, ert=self.ert()
                )
            update_id = self.analyzeStep(run_context, ensemble_id)
            current_iter = self._w_container.iteration_nr - 1

            analysis_success = current_iter > pre_analysis_iter_num
            if analysis_success:
                run_context = self.create_context(
                    current_iter, prior_context=run_context
                )
                ensemble_id = self._runAndPostProcess(
                    run_context, evaluator_server_config, update_id
                )
                num_retries = 0
            else:
                run_context = self.create_context(
                    current_iter, prior_context=run_context, rerun=True
                )
                ensemble_id = self._runAndPostProcess(
                    run_context, evaluator_server_config, update_id
                )
                num_retries += 1

        if current_iter == (phase_count - 1):
            self.setPhase(phase_count, "Simulations completed.")
        else:
            raise ErtRunError(
                (
                    "Iterated ensemble smoother stopped: "
                    "maximum number of iteration retries "
                    f"({num_retries_per_iteration} retries) reached "
                    f"for iteration {current_iter}"
                )
            )

        return run_context

    def create_context(
        self,
        itr: int,
        prior_context: Optional[RunContext] = None,
        rerun: bool = False,
    ) -> RunContext:
        target_case_format = self._simulation_arguments["target_case"]

        sim_fs = self.createTargetCaseFileSystem(itr, target_case_format)

        if prior_context is None:
            mask = self._simulation_arguments["active_realizations"]
        else:
            state: RealizationStateEnum = (
                RealizationStateEnum.STATE_HAS_DATA  # type: ignore
                | RealizationStateEnum.STATE_INITIALIZED
            )
            mask = sim_fs.getStateMap().createMask(state)

        if rerun:
            target_fs = None
        else:
            target_fs = self.createTargetCaseFileSystem(itr + 1, target_case_format)

        run_context = self.ert().create_ensemble_smoother_run_context(
            source_filesystem=sim_fs,
            target_filesystem=target_fs,
            active_mask=mask,
            iteration=itr,
        )
        self._run_context = run_context
        self._last_run_iteration = run_context.iteration
        self.ert().getEnkfFsManager().switchFileSystem(sim_fs)
        return run_context

    @classmethod
    def name(cls) -> str:
        return "Iterated ensemble smoother"
