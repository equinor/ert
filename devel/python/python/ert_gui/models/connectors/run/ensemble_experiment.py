from ert_gui.models import ErtConnector
from ert_gui.models.connectors.run import ActiveRealizationsModel
from ert_gui.models.mixins import RunModelMixin


class EnsembleExperiment(ErtConnector, RunModelMixin):

    def startSimulations(self):
        self.setPhase(0)
        active_realization_mask = ActiveRealizationsModel().getActiveRealizationsMask()
        self.ert().getEnkfSimulationRunner().runEnsembleExperiment(active_realization_mask)
        self.setPhase(1) # done...

    def killAllSimulations(self):
        job_queue = self.ert().siteConfig().getJobQueue()
        job_queue.killAllJobs()

    def __str__(self):
        return "Ensemble Experiment"



