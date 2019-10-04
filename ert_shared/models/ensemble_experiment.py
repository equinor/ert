from res.enkf.enums import HookRuntime
from res.enkf import ErtRunContext

from ert_shared.models import BaseRunModel
from ert_shared import ERT

class EnsembleExperiment(BaseRunModel):

    def __init__(self):
        super(EnsembleExperiment, self).__init__(ERT.enkf_facade.get_queue_config())

    def runSimulations__(self, arguments, run_msg):

        self._job_queue = self._queue_config.create_job_queue( )
        run_context = self.create_context( arguments )

        self.setPhase(0, "Running simulations...", indeterminate=False)

        self.setPhaseName("Pre processing...", indeterminate=True)
        ERT.enkf_facade.create_runpath(run_context)
        ERT.enkf_facade.run_workflows(HookRuntime.PRE_SIMULATION)

        self.setPhaseName( run_msg, indeterminate=False)

        num_successful_realizations = ERT.enkf_facade.run_ensemble_experiment(self._job_queue, run_context)

        num_successful_realizations += arguments.get('prev_successful_realizations', 0)
        self.checkHaveSufficientRealizations(num_successful_realizations)

        self.setPhaseName("Post processing...", indeterminate=True)
        ERT.enkf_facade.run_workflows(HookRuntime.POST_SIMULATION)
        self.setPhase(1, "Simulations completed.") # done...

        return run_context


    def runSimulations(self, arguments ):
        return self.runSimulations__(  arguments , "Running ensemble experiment...")


    def create_context(self, arguments):
        result_fs = ERT.enkf_facade.get_current_file_system()

        runpath_fmt = ERT.enkf_facade.get_runpath_format()
        jobname_fmt = ERT.enkf_facade.get_jobname_format()
        subst_list = ERT.enkf_facade.get_data_kw()
        itr = 0
        mask = arguments["active_realizations"]

        run_context = ErtRunContext.ensemble_experiment(result_fs,
                                                        mask,
                                                        runpath_fmt,
                                                        jobname_fmt,
                                                        subst_list,
                                                        itr)
        self._run_context = run_context
        self._last_run_iteration = run_context.get_iter()
        return run_context

    @classmethod
    def name(cls):
        return "Ensemble Experiment"
