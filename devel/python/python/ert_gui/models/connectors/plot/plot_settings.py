import os
from ert_gui.models import ErtConnector
from ert_gui.models.connectors.init.case_selector import CaseSelectorModel


class PlotSettingsModel(ErtConnector):

    def getDefaultPlotPath(self):
        """ @rtype: str """
        path = self.ert().plotConfig().getPath()
        case_name = CaseSelectorModel().getCurrentChoice()
        path = os.path.join(path, case_name)
        if not os.path.exists(path):
            os.makedirs(path)
        return path


