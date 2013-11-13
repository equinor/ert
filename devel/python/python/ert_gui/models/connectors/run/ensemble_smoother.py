from ert_gui.models import ErtConnector
from ert_gui.models.mixins import ButtonModelMixin, RunModelMixin


class EnsembleSmoother(ErtConnector, RunModelMixin, ButtonModelMixin):

    def startSimulations(self):
        print("Running: %s" % self)

    def killAllSimulations(self):
        print("Kill All %s" % self)

    def __str__(self):
        return "Ensemble Smoother"



