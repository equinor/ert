from ert_gui.models import ErtConnector
from ert_gui.models.mixins import BasicModelMixin


class IteratedRunPathModel(ErtConnector, BasicModelMixin):
    def getValue(self):
        """ @rtype: str """
        return self.ert().analysisConfig().getAnalysisIterConfig().getRunpathFormat()

    def setValue(self, run_path):
        pass






