from ert_shared.feature_toggling import FeatureToggling
from res.enkf.enums import HookRuntime
from res.enkf.enums import RealizationStateEnum
from res.enkf import ErtRunContext, EnkfSimulationRunner
from ert_shared.models import BaseRunModel, ErtRunError
from ert_shared import ERT


class EnsembleSmoother(BaseRunModel):
    def __init__(self):
        super(EnsembleSmoother, self).__init__(
            ERT.enkf_facade.get_queue_config(), phase_count=2
        )
        self.support_restart = False

    def setAnalysisModule(self, module_name):
        module_load_success = self.ert().analysisConfig().selectModule(module_name)

        if not module_load_success:
            raise ErtRunError("Unable to load analysis module '%s'!" % module_name)

    def runSimulations(self, arguments):
        prior_context = self.create_context(arguments)

        self.checkMinimumActiveRealizations(prior_context)
        self.setPhase(0, "Running simulations...", indeterminate=False)

        # self.setAnalysisModule(arguments["analysis_module"])

        self.setPhaseName("Pre processing...", indeterminate=True)
        self.ert().getEnkfSimulationRunner().createRunPath(prior_context)

        EnkfSimulationRunner.runWorkflows(HookRuntime.PRE_SIMULATION, ert=ERT.ert)

        # Push ensemble, parameters, observations to new storage
        ensemble_id = self._post_ensemble_data()

        self.setPhaseName("Running forecast...", indeterminate=False)

        if FeatureToggling.is_enabled("ensemble-evaluator"):
            ee_config = arguments["ee_config"]
            num_successful_realizations = self.run_ensemble_evaluator(
                prior_context, ee_config
            )
        else:
            self._job_queue = self._queue_config.create_job_queue()
            num_successful_realizations = (
                self.ert()
                .getEnkfSimulationRunner()
                .runSimpleStep(self._job_queue, prior_context)
            )

        # Push simulation results to storage
        self._post_ensemble_results(ensemble_id)

        self.checkHaveSufficientRealizations(num_successful_realizations)

        self.setPhaseName("Post processing...", indeterminate=True)
        EnkfSimulationRunner.runWorkflows(HookRuntime.POST_SIMULATION, ert=ERT.ert)

        self.setPhaseName("Analyzing...")
        EnkfSimulationRunner.runWorkflows(HookRuntime.PRE_FIRST_UPDATE, ert=ERT.ert)
        EnkfSimulationRunner.runWorkflows(HookRuntime.PRE_UPDATE, ert=ERT.ert)
        es_update = self.ert().getESUpdate()
        success = es_update.smootherUpdate(prior_context)
        if not success:
            raise ErtRunError("Analysis of simulation failed!")
        EnkfSimulationRunner.runWorkflows(HookRuntime.POST_UPDATE, ert=ERT.ert)

        # Create an update object in storage
        analysis_module_name = self.ert().analysisConfig().activeModuleName()
        update_id = self._post_update_data(ensemble_id, analysis_module_name)

        self.setPhase(1, "Running simulations...")
        self.ert().getEnkfFsManager().switchFileSystem(prior_context.get_target_fs())

        self.setPhaseName("Pre processing...")

        rerun_context = self.create_context(arguments, prior_context=prior_context)

        self.ert().getEnkfSimulationRunner().createRunPath(rerun_context)

        EnkfSimulationRunner.runWorkflows(HookRuntime.PRE_SIMULATION, ert=ERT.ert)
        # Push ensemble, parameters, observations to new storage
        ensemble_id = self._post_ensemble_data(update_id=update_id)

        self.setPhaseName("Running forecast...", indeterminate=False)

        if FeatureToggling.is_enabled("ensemble-evaluator"):
            ee_config = arguments["ee_config"]
            num_successful_realizations = self.run_ensemble_evaluator(
                rerun_context, ee_config
            )
        else:
            self._job_queue = self._queue_config.create_job_queue()
            num_successful_realizations = (
                self.ert()
                .getEnkfSimulationRunner()
                .runSimpleStep(self._job_queue, rerun_context)
            )

        self.checkHaveSufficientRealizations(num_successful_realizations)

        self.setPhaseName("Post processing...", indeterminate=True)
        EnkfSimulationRunner.runWorkflows(HookRuntime.POST_SIMULATION, ert=ERT.ert)

        # Push simulation results to storage
        self._post_ensemble_results(ensemble_id)

        self.setPhase(2, "Simulations completed.")

        return prior_context

    def create_context(self, arguments, prior_context=None):

        model_config = self.ert().getModelConfig()
        runpath_fmt = model_config.getRunpathFormat()
        jobname_fmt = model_config.getJobnameFormat()
        subst_list = self.ert().getDataKW()
        fs_manager = self.ert().getEnkfFsManager()
        if prior_context is None:
            sim_fs = fs_manager.getCurrentFileSystem()
            target_fs = fs_manager.getFileSystem(arguments["target_case"])
            itr = 0
            mask = arguments["active_realizations"]
        else:
            itr = 1
            sim_fs = prior_context.get_target_fs()
            target_fs = None
            state = (
                RealizationStateEnum.STATE_HAS_DATA
                | RealizationStateEnum.STATE_INITIALIZED
            )
            mask = sim_fs.getStateMap().createMask(state)

        # Deleting a run_context removes the possibility to retrospectively
        # determine detailed progress. Thus, before deletion, the detailed
        # progress is stored.
        self.updateDetailedProgress()

        run_context = ErtRunContext.ensemble_smoother(
            sim_fs, target_fs, mask, runpath_fmt, jobname_fmt, subst_list, itr
        )
        self._run_context = run_context
        self._last_run_iteration = run_context.get_iter()
        return run_context

    @classmethod
    def name(cls):
        return "Ensemble Smoother"
