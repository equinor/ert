from res.enkf.enums import EnkfInitModeEnum
from res.enkf.enums import HookRuntime
from res.enkf.enums import RealizationStateEnum
from res.enkf import ErtRunContext
from ert_gui.simulation.models import BaseRunModel, ErtRunError


class EnsembleSmoother(BaseRunModel):

    def __init__(self, queue_config):
        super(EnsembleSmoother, self).__init__("Ensemble Smoother", queue_config , phase_count=2)

    def setAnalysisModule(self, module_name):
        module_load_success = self.ert().analysisConfig().selectModule(module_name)

        if not module_load_success:
            raise ErtRunError("Unable to load analysis module '%s'!" % module_name)


    def runSimulations(self, arguments):
        prior_context = self.create_context( arguments )

        self.checkMinimumActiveRealizations(prior_context)
        self.setPhase(0, "Running simulations...", indeterminate=False)

        # self.setAnalysisModule(arguments["analysis_module"])

        self.setPhaseName("Pre processing...", indeterminate=True)
        self.ert().getEnkfSimulationRunner().createRunPath(prior_context)
        self.ert().getEnkfSimulationRunner().runWorkflows( HookRuntime.PRE_SIMULATION )

        self.setPhaseName("Running forecast...", indeterminate=False)
        self._job_queue = self._queue_config.create_job_queue( )
        num_successful_realizations = self.ert().getEnkfSimulationRunner().runSimpleStep(self._job_queue, prior_context)

        self.checkHaveSufficientRealizations(num_successful_realizations)

        self.setPhaseName("Post processing...", indeterminate=True)
        self.ert().getEnkfSimulationRunner().runWorkflows( HookRuntime.POST_SIMULATION )

        self.setPhaseName("Analyzing...")

        self.ert().getEnkfSimulationRunner().runWorkflows( HookRuntime.PRE_UPDATE )
        es_update = self.ert().getESUpdate( )
        success = es_update.smootherUpdate( prior_context )
        if not success:
            raise ErtRunError("Analysis of simulation failed!")
        self.ert().getEnkfSimulationRunner().runWorkflows( HookRuntime.POST_UPDATE )

        self.setPhase(1, "Running simulations...")
        self.ert().getEnkfFsManager().switchFileSystem( prior_context.get_target_fs( ) )

        self.setPhaseName("Pre processing...")

        rerun_context = self.create_context( arguments, prior_context = prior_context )
        self.ert().getEnkfSimulationRunner().createRunPath( rerun_context )
        self.ert().getEnkfSimulationRunner().runWorkflows( HookRuntime.PRE_SIMULATION )

        self.setPhaseName("Running forecast...", indeterminate=False)

        self._job_queue = self._queue_config.create_job_queue( )
        num_successful_realizations = self.ert().getEnkfSimulationRunner().runSimpleStep(self._job_queue, rerun_context)

        self.checkHaveSufficientRealizations(num_successful_realizations)

        self.setPhaseName("Post processing...", indeterminate=True)
        self.ert().getEnkfSimulationRunner().runWorkflows( HookRuntime.POST_SIMULATION )

        self.setPhase(2, "Simulations completed.")
        self.__job_queue = None


    def create_context(self, arguments, prior_context = None):
        model_config = self.ert().getModelConfig( )
        runpath_fmt = model_config.getRunpathFormat( )
        jobname_fmt = model_config.getJobnameFormat( )
        subst_list = self.ert().getDataKW( )
        fs_manager = self.ert().getEnkfFsManager()
        if prior_context is None:
            sim_fs = fs_manager.getCurrentFileSystem( )
            target_fs = fs_manager.getFileSystem(arguments["target_case"])
            itr = 0
            mask = arguments["active_realizations"]
        else:
            itr = 1
            sim_fs = prior_context.get_target_fs( )
            target_fs = None
            state = RealizationStateEnum.STATE_HAS_DATA | RealizationStateEnum.STATE_INITIALIZED
            mask = sim_fs.getStateMap().createMask(state)

        run_context = ErtRunContext.ensemble_smoother( sim_fs, target_fs, mask, runpath_fmt, jobname_fmt, subst_list, itr)
        return run_context


