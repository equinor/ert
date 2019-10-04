from res.enkf.enums import HookRuntime
from res.enkf.enums import RealizationStateEnum
from res.enkf import ErtRunContext
from ert_shared.models import BaseRunModel, ErtRunError
from ert_shared import ERT

class EnsembleSmoother(BaseRunModel):

    def __init__(self):
        super(EnsembleSmoother, self).__init__(ERT.enkf_facade.get_queue_config() , phase_count=2)
        self.support_restart = False

    def setAnalysisModule(self, module_name):
        return ERT.enkf_facade.set_analysis_module(module_name)


    def runSimulations(self, arguments):
        prior_context = self.create_context( arguments )

        self.checkMinimumActiveRealizations(prior_context)
        self.setPhase(0, "Running simulations...", indeterminate=False)

        # self.setAnalysisModule(arguments["analysis_module"])

        self.setPhaseName("Pre processing...", indeterminate=True)
        ERT.enkf_facade.create_runpath(prior_context)
        ERT.enkf_facade.run_workflows(HookRuntime.PRE_SIMULATION)

        self.setPhaseName("Running forecast...", indeterminate=False)
        self._job_queue = self._queue_config.create_job_queue( )
        num_successful_realizations = ERT.enkf_facade.run_simple_step(self._job_queue, prior_context)

        self.checkHaveSufficientRealizations(num_successful_realizations)

        self.setPhaseName("Post processing...", indeterminate=True)
        ERT.enkf_facade.run_workflows(HookRuntime.POST_SIMULATION)

        self.setPhaseName("Analyzing...")

        ERT.enkf_facade.run_workflows(HookRuntime.PRE_UPDATE)
        success = ERT.enkf_facade.smoother_update(prior_context)
        
        if not success:
            raise ErtRunError("Analysis of simulation failed!")
        ERT.enkf_facade.run_workflows(HookRuntime.POST_UPDATE)

        self.setPhase(1, "Running simulations...")
        ERT.enkf_facade.switch_file_system(prior_context.get_target_fs())

        self.setPhaseName("Pre processing...")

        rerun_context = self.create_context( arguments, prior_context = prior_context )

        ERT.enkf_facade.create_runpath(rerun_context )
        ERT.enkf_facade.run_workflows(HookRuntime.PRE_SIMULATION)

        self.setPhaseName("Running forecast...", indeterminate=False)

        self._job_queue = self._queue_config.create_job_queue( )
        num_successful_realizations = ERT.enkf_facade.run_simple_step(self._job_queue, rerun_context)

        self.checkHaveSufficientRealizations(num_successful_realizations)

        self.setPhaseName("Post processing...", indeterminate=True)
        ERT.enkf_facade.run_workflows(HookRuntime.POST_SIMULATION)

        self.setPhase(2, "Simulations completed.")

        return prior_context


    def create_context(self, arguments, prior_context = None):
        runpath_fmt = ERT.enkf_facade.get_runpath_format()
        jobname_fmt = ERT.enkf_facade.get_jobname_format()
        subst_list = ERT.enkf_facade.get_data_kw()
        if prior_context is None:
            sim_fs = ERT.enkf_facade.get_current_file_system()
            target_fs = ERT.enkf_facade.get_file_system(arguments["target_case"])
            itr = 0
            mask = arguments["active_realizations"]
        else:
            itr = 1
            sim_fs = prior_context.get_target_fs()
            target_fs = None
            state = RealizationStateEnum.STATE_HAS_DATA | RealizationStateEnum.STATE_INITIALIZED
            mask = sim_fs.getStateMap().createMask(state)

        run_context = ErtRunContext.ensemble_smoother(sim_fs, target_fs, mask, runpath_fmt, jobname_fmt, subst_list, itr)
        self._run_context = run_context
        self._last_run_iteration = run_context.get_iter()
        return run_context

    @classmethod
    def name(cls):
        return "Ensemble Smoother"
