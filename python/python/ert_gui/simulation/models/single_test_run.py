
from ert_gui.simulation.models import BaseRunModel, ErtRunError, EnsembleExperiment
from ert_gui.ertwidgets.models.ertmodel import getRealizationCount, getRunPath, getQueueConfig

class SingleTestRun(EnsembleExperiment):

    def __init__(self):
        queue_config = getQueueConfig()
        local_queue_config = queue_config.create_local_copy()
        super(EnsembleExperiment, self).__init__( local_queue_config)

    def checkHaveSufficientRealizations(self, num_successful_realizations):
        #Should only have one successful realization
        if num_successful_realizations == 0:
            raise ErtRunError("Simulation failed!")

    def runSimulations(self, arguments):
        return self.runSimulations__( arguments  , "Running single realisation test ...")

    @classmethod
    def name(cls):
        return "Single realization test-run"
