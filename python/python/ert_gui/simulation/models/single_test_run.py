from ecl.util.util import BoolVector
from res.enkf.enums import HookRuntime
from res.enkf import ErtRunContext
from ert_gui.simulation.models import BaseRunModel, ErtRunError, EnsembleExperiment

class SingleTestRun(BaseRunModel):

    def __init__(self, queue_config):
        super(SingleTestRun, self).__init__("Single realization test-run" , queue_config)



    def runSimulations(self, arguments):
        run_msg =  "Running single realisation test ...";
        self._job_queue = self._queue_config.create_job_queue( )
        run_context = self.create_context( arguments )
        self.setPhase(0, "Running simulations...", indeterminate=False)

        self.setPhaseName("Pre processing...", indeterminate=True)
        self.ert().getEnkfSimulationRunner().createRunPath( run_context )
        self.ert().getEnkfSimulationRunner().runWorkflows( HookRuntime.PRE_SIMULATION )

        self.setPhaseName( run_msg, indeterminate=False)

        num_successful_realizations = self.ert().getEnkfSimulationRunner().runEnsembleExperiment(self._job_queue, run_context)
        self.checkHaveSufficientRealizations(num_successful_realizations)

        self.setPhaseName("Post processing...", indeterminate=True)
        self.ert().getEnkfSimulationRunner().runWorkflows( HookRuntime.POST_SIMULATION )
        self.setPhase(1, "Simulations completed.") # done...
        self._job_queue = None

    def checkHaveSufficientRealizations(self, num_successful_realizations):
        #Should only have one successful realization
        if num_successful_realizations == 0:
            raise ErtRunError("Simulation failed!")


    def create_context(self, arguments):
        fs_manager = self.ert().getEnkfFsManager()
        result_fs = fs_manager.getCurrentFileSystem( )

        model_config = self.ert().getModelConfig( )
        runpath_fmt = model_config.getRunpathFormat( )
        jobname_fmt = model_config.getJobnameFormat( )
        subst_list = self.ert().getDataKW( )
        itr = 0

        mask = BoolVector(  default_value = False )
        mask[0] = True
        run_context = ErtRunContext.ensemble_experiment(result_fs,
                                                        mask,
                                                        runpath_fmt,
                                                        jobname_fmt,
                                                        subst_list,
                                                        itr)
        return run_context
