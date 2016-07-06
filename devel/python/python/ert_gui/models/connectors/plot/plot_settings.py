import os
from ert_gui.models import ErtConnector

class PlotSettingsModel(ErtConnector):

    def getDefaultPlotPath(self):
        """ @rtype: str """
        path = self.ert().plotConfig().getPath()
        if not os.path.exists(path):
            os.makedirs(path)
        return path


