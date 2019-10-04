from res.enkf.enums import HookRuntime
from res.enkf import ErtRunContext
from ert_shared.models import BaseRunModel, ErtRunError
from ert_shared import ERT

class IteratedEnsembleSmoother(BaseRunModel):

    def __init__(self):
        super(IteratedEnsembleSmoother, self).__init__(ERT.enkf_facade.get_queue_config() , phase_count=2)
        self.support_restart = False

    def setAnalysisModule(self, module_name):
        return ERT.enkf_facade.set_analysis_module(module_name)

    def _runAndPostProcess(self, run_context):
        self._job_queue = self._queue_config.create_job_queue( )
        phase_msg = "Running iteration %d of %d simulation iterations..." % (run_context.get_iter(), self.phaseCount() - 1)
        self.setPhase(run_context.get_iter(), phase_msg, indeterminate=False)

        self.setPhaseName("Pre processing...", indeterminate=True)
        ERT.enkf_facade.create_runpath(run_context)
        ERT.enkf_facade.run_workflows(HookRuntime.PRE_SIMULATION)

        self.setPhaseName("Running forecast...", indeterminate=False)
        num_successful_realizations = ERT.enkf_facade.run_simple_step(self._job_queue, run_context)

        self.checkHaveSufficientRealizations(num_successful_realizations)

        self.setPhaseName("Post processing...", indeterminate=True)
        ERT.enkf_facade.run_workflows(HookRuntime.POST_SIMULATION)


    def createTargetCaseFileSystem(self, phase, target_case_format):
        return ERT.enkf_facade.get_file_system(target_case_format % phase)

    def analyzeStep(self, run_context):        
        self.setPhaseName("Analyzing...", indeterminate=True)        
        self.setPhaseName("Pre processing update...", indeterminate=True)
        ERT.enkf_facade.run_workflows(HookRuntime.PRE_UPDATE)
        success = ERT.enkf_facade.smoother_update(run_context)

        if not success:
            raise ErtRunError("Analysis of simulation failed!")

        self.setPhaseName("Post processing update...", indeterminate=True)
        ERT.enkf_facade.run_workflows(HookRuntime.POST_UPDATE)

    def runSimulations(self, arguments):
        phase_count = ERT.enkf_facade.get_number_of_iterations() + 1
        self.setPhaseCount(phase_count)

        analysis_module = self.setAnalysisModule(arguments["analysis_module"])
        target_case_format = arguments["target_case"]
        run_context = self.create_context( arguments , 0 )

        ERT.enkf_facade.set_case_format(target_case_format)
        
        self._runAndPostProcess( run_context )

        num_retries_per_iteration = ERT.enkf_facade.get_number_of_retries()
        number_of_iterations = ERT.enkf_facade.get_number_of_iterations()
        num_retries = 0
        current_iter = 0

        while current_iter < number_of_iterations and num_retries < num_retries_per_iteration:
            pre_analysis_iter_num = analysis_module.getInt("ITER")
            self.analyzeStep( run_context )
            current_iter = analysis_module.getInt("ITER")

            analysis_success = current_iter > pre_analysis_iter_num
            if analysis_success:
                run_context = self.create_context( arguments, current_iter, prior_context = run_context )
                ERT.enkf_facade.switch_file_system(run_context.get_target_fs())
                self._runAndPostProcess(run_context)
                num_retries = 0
            else:
                run_context = self.create_context( arguments, current_iter, prior_context = run_context , rerun = True)
                self._runAndPostProcess(run_context)
                num_retries += 1

        if current_iter == (phase_count - 1):
            self.setPhase(phase_count, "Simulations completed.")
        else:
            raise ErtRunError("Iterated Ensemble Smoother stopped: maximum number of iteration retries (%d retries) reached for iteration %d" % (num_retries_per_iteration, current_iter))

        return run_context


    def create_context(self, arguments, itr, prior_context = None, rerun = False):        
        runpath_fmt = ERT.enkf_facade.get_runpath_format()
        jobname_fmt = ERT.enkf_facade.get_jobname_format()
        subst_list = ERT.enkf_facade.get_data_kw()
        target_case_format = arguments["target_case"]

        if prior_context is None:
            mask = arguments["active_realizations"]
        else:
            mask = prior_context.get_mask( )

        sim_fs = self.createTargetCaseFileSystem(itr, target_case_format)
        if rerun:
            target_fs = None
        else:
            target_fs = self.createTargetCaseFileSystem(itr + 1 , target_case_format)

        run_context = ErtRunContext.ensemble_smoother( sim_fs, target_fs, mask, runpath_fmt, jobname_fmt, subst_list, itr)
        self._run_context = run_context
        self._last_run_iteration = run_context.get_iter()
        return run_context

    @classmethod
    def name(cls):
        return "Iterated Ensemble Smoother - Experimental"
