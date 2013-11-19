import time
from ert_gui.models import ErtConnector
from ert_gui.models.connectors.run import NumberOfIterationsModel
from ert_gui.models.mixins import RunModelMixin


class IteratedEnsembleSmoother(ErtConnector, RunModelMixin):

    def __init__(self):
        super(IteratedEnsembleSmoother, self).__init__(phase_count=2)

    def startSimulations(self):

        phase_count = NumberOfIterationsModel().getValue()
        self.setPhaseCount(phase_count)

        phase = 0
        for phase in range(self.phaseCount()):
            self.setIndeterminate(False)
            self.setPhase(phase, "Running iteration %d of %d simulation iterations..." % (phase + 1, phase_count))
            time.sleep(5)

            self.setPhaseName("Analyzing...")
            self.setIndeterminate(True)
            time.sleep(5)

        self.setPhase(phase_count, "Simulations completed.")


    def killAllSimulations(self):
        job_queue = self.ert().siteConfig().getJobQueue()
        job_queue.killAllJobs()

    def __str__(self):
        return "Iterated Ensemble Smoother"


