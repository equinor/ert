from ert_gui.models import ErtConnector
from ert_gui.models.mixins import ListModelMixin


class CaseList(ErtConnector, ListModelMixin):

    def getList(self):
        fs = self.ert().getEnkfFsManager().getFileSystem()
        case_list = self.ert().getEnkfFsManager().getCaseList()
        return sorted(case_list)

    def addItem(self, value):
        self.ert().getEnkfFsManager().selectFileSystem(value)
        self.observable().notify(ListModelMixin.LIST_CHANGED_EVENT)

    # def removeItem(self, value):
    #     print("Remove item: %s" % value)
    #     pass







