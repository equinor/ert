from typing import Optional, Dict, Any

from res.enkf import ErtRunContext, EnkfSimulationRunner
from res.enkf.enkf_fs import EnkfFs
from res.enkf.enums import HookRuntime, RealizationStateEnum
from res.enkf.enkf_main import EnKFMain, QueueConfig
from res.analysis.analysis_module import AnalysisModule

from ert_shared.models import BaseRunModel, ErtRunError
from ert_shared.ensemble_evaluator.config import EvaluatorServerConfig


class IteratedEnsembleSmoother(BaseRunModel):
    def __init__(
        self,
        simulation_arguments: Dict[str, Any],
        ert: EnKFMain,
        queue_config: QueueConfig,
    ):
        super().__init__(simulation_arguments, ert, queue_config, phase_count=2)
        self.support_restart = False

    def setAnalysisModule(self, module_name: str) -> AnalysisModule:
        module_load_success = self.ert().analysisConfig().selectModule(module_name)

        if not module_load_success:
            raise ErtRunError("Unable to load analysis module '%s'!" % module_name)

        return self.ert().analysisConfig().getModule(module_name)

    def _runAndPostProcess(
        self,
        run_context: ErtRunContext,
        evaluator_server_config: EvaluatorServerConfig,
        update_id: Optional[str] = None,
    ) -> str:
        phase_msg = "Running iteration %d of %d simulation iterations..." % (
            run_context.get_iter(),
            self.phaseCount() - 1,
        )
        self.setPhase(run_context.get_iter(), phase_msg, indeterminate=False)

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

    def analyzeStep(self, run_context: ErtRunContext, ensemble_id: str) -> str:
        self.setPhaseName("Analyzing...", indeterminate=True)

        self.setPhaseName("Pre processing update...", indeterminate=True)
        EnkfSimulationRunner.runWorkflows(HookRuntime.PRE_UPDATE, ert=self.ert())
        es_update = self.ert().getESUpdate()

        success = es_update.smootherUpdate(run_context)

        # Push update data to new storage
        analysis_module_name = self.ert().analysisConfig().activeModuleName()
        update_id = self._post_update_data(
            parent_ensemble_id=ensemble_id, algorithm=analysis_module_name
        )

        if not success:
            raise ErtRunError("Analysis of simulation failed!")

        self.setPhaseName("Post processing update...", indeterminate=True)
        EnkfSimulationRunner.runWorkflows(HookRuntime.POST_UPDATE, ert=self.ert())
        return update_id

    def runSimulations(
        self, evaluator_server_config: EvaluatorServerConfig
    ) -> ErtRunContext:
        phase_count = self.facade.get_number_of_iterations() + 1
        self.setPhaseCount(phase_count)

        analysis_module = self.setAnalysisModule(
            self._simulation_arguments["analysis_module"]
        )
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
            pre_analysis_iter_num = analysis_module.getInt("ITER")
            # We run the PRE_FIRST_UPDATE hook here because the current_iter is
            # explicitly available, versus in the run_context inside analyzeStep
            if current_iter == 0:
                EnkfSimulationRunner.runWorkflows(
                    HookRuntime.PRE_FIRST_UPDATE, ert=self.ert()
                )
            update_id = self.analyzeStep(run_context, ensemble_id)
            current_iter = analysis_module.getInt("ITER")

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
                    "(%d retries) reached for iteration %d"
                )
                % (num_retries_per_iteration, current_iter)
            )

        return run_context

    def create_context(
        self,
        itr: int,
        prior_context: Optional[ErtRunContext] = None,
        rerun: bool = False,
    ) -> ErtRunContext:
        model_config = self.ert().getModelConfig()
        runpath_fmt = model_config.getRunpathFormat()
        jobname_fmt = model_config.getJobnameFormat()
        subst_list = self.ert().getDataKW()
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

        run_context = ErtRunContext.ensemble_smoother(
            sim_fs, target_fs, mask, runpath_fmt, jobname_fmt, subst_list, itr
        )
        self._run_context = run_context
        self._last_run_iteration = run_context.get_iter()
        self.ert().getEnkfFsManager().switchFileSystem(sim_fs)
        return run_context

    @classmethod
    def name(cls) -> str:
        return "Iterated ensemble smoother"
