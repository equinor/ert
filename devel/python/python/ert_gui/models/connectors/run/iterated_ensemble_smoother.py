import time
from ert.enkf.enums import EnkfInitModeEnum
from ert_gui.models import ErtConnector
from ert_gui.models.connectors.run import ActiveRealizationsModel, TargetCaseModel, AnalysisModuleModel
from ert_gui.models.mixins import RunModelMixin


class IteratedEnsembleSmoother(ErtConnector, RunModelMixin):

    def __init__(self):
        super(IteratedEnsembleSmoother, self).__init__(phase_count=2)

    def startSimulations(self):
        self.setPhase(0)


        self.setPhase(2)


    def killAllSimulations(self):
        job_queue = self.ert().siteConfig().getJobQueue()
        job_queue.killAllJobs()

    def __str__(self):
        return "Iterated Ensemble Smoother"
