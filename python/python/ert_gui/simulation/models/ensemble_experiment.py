from res.enkf.enums import HookRuntime
from ert_gui.simulation.models import BaseRunModel, ErtRunError

class EnsembleExperiment(BaseRunModel):

    def __init__(self):
        super(EnsembleExperiment, self).__init__("Ensemble Experiment")

    def runSimulations__(self, job_queue,  active_realization_mask, run_msg):
        self.setPhase(0, "Running simulations...", indeterminate=False)

        active_realizations = self.count_active_realizations( active_realization_mask )
        self.setPhaseName("Pre processing...", indeterminate=True)
        self.ert().getEnkfSimulationRunner().createRunPath(active_realization_mask, 0)
        self.ert().getEnkfSimulationRunner().runWorkflows( HookRuntime.PRE_SIMULATION )

        self.setPhaseName( run_msg, indeterminate=False)

        num_successful_realizations = self.ert().getEnkfSimulationRunner().runEnsembleExperiment(job_queue, active_realization_mask)
        self.assertHaveSufficientRealizations(num_successful_realizations, active_realizations)

        self.setPhaseName("Post processing...", indeterminate=True)
        self.ert().getEnkfSimulationRunner().runWorkflows( HookRuntime.POST_SIMULATION )
        self.setPhase(1, "Simulations completed.") # done...

        
        
    def runSimulations(self, job_queue,  arguments):
        active_realization_mask = arguments["active_realizations"]
        self.runSimulations__( job_queue , active_realizations_mask , "Running ensemble experiment...")
                
        

        
    def count_active_realizations(self, active_realization_mask):
        return sum(active_realization_mask)

