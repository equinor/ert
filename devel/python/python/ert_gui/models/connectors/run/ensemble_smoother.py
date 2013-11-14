import time
from ert_gui.models import ErtConnector
from ert_gui.models.mixins import RunModelMixin


class EnsembleSmoother(ErtConnector, RunModelMixin):

    def __init__(self):
        super(EnsembleSmoother, self).__init__(phase_count=2)

    def startSimulations(self):
        time.sleep(10)
        self.setPhase(1)
        time.sleep(5)
        self.setPhase(2)


    def killAllSimulations(self):
        print("Kill All %s" % self)

    def __str__(self):
        return "Ensemble Smoother"



