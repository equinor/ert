from res.enkf.enums import HookRuntime
from ert_gui.simulation.models import BaseRunModel, ErtRunError

class SingleTestRun(BaseRunModel):
    
    def __init__(self):
        super(SingleTestRun, self).__init__("Single test-run")

    def runSimulations(self, job_queue,  arguments):
        self.setPhase(0, "Running simulations...", indeterminate=False)
        active_realization_mask = arguments["active_realizations"]

        active_realizations = 1;
        for n in range(1, len(active_realization_mask)):
            active_realization_mask[n] = False

        active_realization_mask[0] = True;

        self.setPhaseName("Pre processing...", indeterminate=True)
        self.ert().getEnkfSimulationRunner().createRunPath(active_realization_mask, 0)
        self.ert().getEnkfSimulationRunner().runWorkflows( HookRuntime.PRE_SIMULATION )

        self.setPhaseName("Running ensemble experiment...", indeterminate=False)

        num_successful_realizations = self.ert().getEnkfSimulationRunner().runEnsembleExperiment(job_queue, active_realization_mask)

        self.assertHaveSufficientRealizations(num_successful_realizations, active_realizations)

        self.setPhaseName("Post processing...", indeterminate=True)
        self.ert().getEnkfSimulationRunner().runWorkflows( HookRuntime.POST_SIMULATION )

        self.setPhase(1, "Simulations completed.") # done...

