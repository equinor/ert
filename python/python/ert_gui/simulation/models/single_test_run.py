from res.enkf.enums import HookRuntime
from ert_gui.simulation.models import BaseRunModel, ErtRunError, EnsembleExperiment

class SingleTestRun(EnsembleExperiment):
    
    def __init__(self):
        super(EnsembleExperiment, self).__init__("Single test-run")

    def count_active_realizations(self, active_realization_mask):
        for n in range(1, len(active_realization_mask)):
            active_realization_mask[n] = False

        active_realization_mask[0] = True;
        return 1

    

