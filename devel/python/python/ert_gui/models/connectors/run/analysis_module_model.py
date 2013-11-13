from ert_gui.models import ErtConnector
from ert_gui.models.mixins import ChoiceModelMixin


class AnalysisModuleModel(ErtConnector, ChoiceModelMixin):

    def __init__(self):
        self.__value = None
        super(AnalysisModuleModel, self).__init__()

    def getChoices(self):
        return sorted(self.ert().analysisConfig().getModuleList())

    def getCurrentChoice(self):
        if self.__value is None:
            active_name = self.ert().analysisConfig().activeModuleName()
            modules = self.getChoices()
            if active_name in modules:
                self.__value = active_name
            else:
                self.__value = modules[0]
        return self.__value

    def setCurrentChoice(self, value):
        self.__value = value
        self.observable().notify(self.CURRENT_CHOICE_CHANGED_EVENT)





