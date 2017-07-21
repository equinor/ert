from res.enkf.enums import HookRuntime
from ert_gui.simulation.models import BaseRunModel, ErtRunError

class EnsembleExperiment(BaseRunModel):

    def __init__(self, queue_config):
        super(EnsembleExperiment, self).__init__("Ensemble Experiment" , queue_config)

    def runSimulations__(self, job_queue, run_context, run_msg):
        self.setPhase(0, "Running simulations...", indeterminate=False)

        active_realizations = self.count_active_realizations( run_context )
        self.setPhaseName("Pre processing...", indeterminate=True)
        self.ert().getEnkfSimulationRunner().createRunPath( run_context )
        self.ert().getEnkfSimulationRunner().runWorkflows( HookRuntime.PRE_SIMULATION )

        self.setPhaseName( run_msg, indeterminate=False)

        num_successful_realizations = self.ert().getEnkfSimulationRunner().runEnsembleExperiment(job_queue, run_context)
        self.assertHaveSufficientRealizations(num_successful_realizations, active_realizations )

        self.setPhaseName("Post processing...", indeterminate=True)
        self.ert().getEnkfSimulationRunner().runWorkflows( HookRuntime.POST_SIMULATION )
        self.setPhase(1, "Simulations completed.") # done...

        
        
    def runSimulations(self, job_queue,  arguments):
        active_realizations_mask = arguments["active_realizations"]
        self.runSimulations__( job_queue , active_realizations_mask , "Running ensemble experiment...")
                
        

        
    def count_active_realizations(self, run_context):
        return sum(run_context.get_mask( ))

