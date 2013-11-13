import time
from ert_gui.models import ErtConnector
from ert_gui.models.mixins import ModelMixin


class RunStatusModel(ErtConnector, ModelMixin):
    STATUS_CHANGED_EVENT = "status_changed_event"

    def __init__(self):
        super(RunStatusModel, self).__init__()

        self.__status = {}
        self.__resetStatusFlag()
        self.__status_count = {}

        self.__is_running = False
        self.job_queue = None


    def registerDefaultEvents(self):
        super(RunStatusModel, self).registerDefaultEvents()
        self.observable().addEvent(RunStatusModel.STATUS_CHANGED_EVENT)


    def __processStatus(self):
        self.__resetStatusFlag()
        for job_number in range(len(self.job_queue)):
            status = self.job_queue.getJobStatus(job_number)
            self.__setMemberStatus(job_number, status)

        self.__updateStatusCount()
        self.__checkStatusChangedFlag() # Emit once for all members if any changes has occurred
        time.sleep(0.5)

    def startStatusPoller(self):
        assert not self.__is_running, "Job already started!"

        self.job_queue = self.ert().siteConfig().getJobQueue()

        while not self.job_queue.isRunning():
            time.sleep(0.5)

        self.__is_running = True

        self.__resetMemberStatus()

        while self.job_queue.isRunning():
            self.__processStatus()

        self.__processStatus() # catch all the latest changes


        self.__is_running = False


    def getStatusCounts(self):
        return dict(self.__status_count)

    def getActiveCount(self):
        return len(self.job_queue)

    def __setMemberStatus(self, job_number, status):
        if not self.__status.has_key(job_number):
            self.__status[job_number] = status
            self.__setStatusChangedFlag()

        if self.__status[job_number] != status:
            self.__status[job_number] = status
            self.__setStatusChangedFlag()

    def __resetStatusFlag(self):
        self.__status_changed_flag = False

    def __setStatusChangedFlag(self):
        self.__status_changed_flag = True

    def __checkStatusChangedFlag(self):
        if self.__status_changed_flag:
            self.observable().notify(RunStatusModel.STATUS_CHANGED_EVENT)

    def __resetMemberStatus(self):
        self.__status.clear()

    def __updateStatusCount(self):
        self.__status_count.clear()

        for job_number in self.__status:
            status = self.__status[job_number]

            if not self.__status_count.has_key(status):
                self.__status_count[status] = 0

            self.__status_count[status] += 1

