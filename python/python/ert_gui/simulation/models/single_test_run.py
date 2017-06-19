from ecl.util import BoolVector
from res.enkf.enums import HookRuntime
from ert_gui.simulation.models import BaseRunModel, ErtRunError, EnsembleExperiment

class SingleTestRun(EnsembleExperiment):
    
    def __init__(self):
        super(EnsembleExperiment, self).__init__("Single realization test-run")


        
    def runSimulations(self, job_queue,  arguments):
        mask = BoolVector(  default_value = False )
        mask[0] = True
        self.runSimulations__( job_queue ,  mask , "Running single realisation test ...")
        

    
    

