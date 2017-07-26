from res.enkf.enums import EnkfInitModeEnum, HookRuntime
from ert_gui.ertwidgets.models.ertmodel import getNumberOfIterations
from ert_gui.simulation.models import BaseRunModel, ErtRunError


class IteratedEnsembleSmoother(BaseRunModel):

    def __init__(self, queue_config):
        super(IteratedEnsembleSmoother, self).__init__("Iterated Ensemble Smoother", queue_config , phase_count=2)

    def setAnalysisModule(self, module_name):
        module_load_success = self.ert().analysisConfig().selectModule(module_name)

        if not module_load_success:
            raise ErtRunError("Unable to load analysis module '%s'!" % module_name)

        return self.ert().analysisConfig().getModule(module_name)


    def _runAndPostProcess(self, run_context):
        self._job_queue = self._queue_config.create_job_queue( )
        self.setPhase(phase, "Running iteration %d of %d simulation iterations..." % (phase, phase_count - 1), indeterminate=False)

        self.setPhaseName("Pre processing...", indeterminate=True)
        self.ert().getEnkfSimulationRunner().createRunPath(active_realization_mask, phase)
        self.ert().getEnkfSimulationRunner().runWorkflows( HookRuntime.PRE_SIMULATION )

        self.setPhaseName("Running forecast...", indeterminate=False)
        num_successful_realizations = self.ert().getEnkfSimulationRunner().runSimpleStep(job_queue, active_realization_mask, mode, phase)

        self.checkHaveSufficientRealizations(num_successful_realizations)

        self.setPhaseName("Post processing...", indeterminate=True)
        self.ert().getEnkfSimulationRunner().runWorkflows( HookRuntime.POST_SIMULATION )
        self._job_queue = None

        

    def createTargetCaseFileSystem(self, phase, target_case_format):
        target_fs = self.ert().getEnkfFsManager().getFileSystem(target_case_format % phase)
        return target_fs


    def analyzeStep(self, run_context):
        target_fs = run_context.get_target_fs( )
        self.setPhaseName("Analyzing...", indeterminate=True)
        source_fs = self.ert().getEnkfFsManager().getCurrentFileSystem()

        self.setPhaseName("Pre processing update...", indeterminate=True)
        self.ert().getEnkfSimulationRunner().runWorkflows(HookRuntime.PRE_UPDATE)
        es_update = self.ert().getESUpdate()

        success = es_update.smootherUpdate(source_fs, target_fs)
        if not success:
            raise ErtRunError("Analysis of simulation failed!")

        self.setPhaseName("Post processing update...", indeterminate=True)
        self.ert().getEnkfSimulationRunner().runWorkflows(HookRuntime.POST_UPDATE)

    def runSimulations(self, arguments):
        phase_count = getNumberOfIterations() + 1
        self.setPhaseCount(phase_count)

        analysis_module = self.setAnalysisModule(arguments["analysis_module"])
        run_context = self.create_context( arguments )
        self.ert().analysisConfig().getAnalysisIterConfig().setCaseFormat( target_case_format )

        self._runAndPostProcess( run_context )


        analysis_config = self.ert().analysisConfig()
        analysis_iter_config = analysis_config.getAnalysisIterConfig()
        num_retries_per_iteration = analysis_iter_config.getNumRetries()
        num_tries = 0
        current_iteration = 1

        while current_iteration <= getNumberOfIterations() and num_tries < num_retries_per_iteration:
            pre_analysis_iter_num = analysis_module.getInt("ITER")
            self.analyzeStep( run_context )
            post_analysis_iter_num = analysis_module.getInt("ITER")

            analysis_success = False
            if  post_analysis_iter_num > pre_analysis_iter_num:
                analysis_success = True


                
            if analysis_success:
                current_iteration += 1
                run_context = self.create_context( arguments, current_iteration, prior_context = run_context )
                self.ert().getEnkfFsManager().switchFileSystem(run_context.get_target_fs())
                self._runAndPostProcess(job_queue, run_context)
                num_tries = 0
            else:
                run_context = self.create_context( arguments, current_iteration, prior_context = run_context , rerun = True)
                self._runAndPostProcess(job_queue, run_context)
                num_tries += 1



        if current_iteration == phase_count:
            self.setPhase(phase_count, "Simulations completed.")
        else:
            raise ErtRunError("Iterated Ensemble Smoother stopped: maximum number of iteration retries (%d retries) reached for iteration %d" % (num_retries_per_iteration, current_iteration))


    def create_context(self, arguments, itr, prior_context = None, rerun = False):
        model_config = self.ert().getModelConfig( )
        runpath_fmt = model_config.getRunpathFormat( )
        subst_list = self.ert().getDataKW( )
        fs_manager = self.ert().getEnkfFsManager()
        target_case_format = arguments["target_case"]
        if prior_context is None:
            source_fs = self.ert().getEnkfFsManager().getCurrentFileSystem()
            initial_fs = self.createTargetCaseFileSystem(0, target_case_format)

            if not source_fs == initial_fs:
                self.ert().getEnkfFsManager().switchFileSystem(initial_fs)
                self.ert().getEnkfFsManager().initializeCurrentCaseFromExisting(source_fs, 0)

            mask = arguments["active_realizations"]
        else:
            mask = prior_context.get_mask( )
            if rerun:
                init_fs = prior_context.get_target_fs( )
                run_context = ErtRunContext( EnkfTunType.SMOOTHER_UPDATE , init_fs, sim_fs, target_fs, mask, runpath_fmt, subst_list, itr)
                return run_context
                
            sim_fs = prior_context.get_target_fs( )
            target_fs = self.createTargetCaseFileSystem(itr, target_case_format)
            
        run_context = ErtRunContext.ensemble_smoother( sim_fs, target_fs, mask, runpath_fmt, subst_list, itr)
        return run_context



