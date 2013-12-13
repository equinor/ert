from ert_gui.models import ErtConnector
from ert_gui.models.mixins import ListModelMixin
from ert.enkf.util import TimeMap


class ReportStepsModel(ErtConnector, ListModelMixin):

    def __init__(self):
        self.__time_map = None
        self.__report_steps = None
        super(ReportStepsModel, self).__init__()


    def getReportSteps(self):
        """ @rtype: TimeMap """
        if self.__time_map is None:
            enkf_fs = self.ert().getEnkfFsManager().getFileSystem()
            self.__time_map = enkf_fs.getTimeMap()

        return self.__time_map


    def getList(self):
        """ @rtype: list of ctime """
        return [c_time for c_time in self.getReportSteps()]



