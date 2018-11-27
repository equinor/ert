from ecl.util.util import BoolVector
from res.enkf.enums import HookRuntime
from res.enkf import ErtRunContext
from ert_gui.simulation.models import BaseRunModel, ErtRunError, EnsembleExperiment

class SingleTestRun(EnsembleExperiment):

    def __init__(self, queue_config):
        super(EnsembleExperiment, self).__init__("Single realization test-run" , queue_config)

    def checkHaveSufficientRealizations(self, num_successful_realizations):
        #Should only have one successful realization
        if num_successful_realizations == 0:
            raise ErtRunError("Simulation failed!")

    def runSimulations(self, arguments):
        return self.runSimulations__( arguments  , "Running single realisation test ...")
