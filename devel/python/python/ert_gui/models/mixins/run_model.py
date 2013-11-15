import time
from ert.job_queue.job_status_type_enum import JobStatusType
from ert_gui.models.mixins import ModelMixin, AbstractMethodError


class RunModelMixin(ModelMixin):
    RUN_PHASE_CHANGED_EVENT = "run_phase_changed_event"
    RUN_FAILED_EVENT = "run_failed_event"
    RUN_FINISHED_EVENT = "run_finished_event"

    def registerDefaultEvents(self):
        super(RunModelMixin, self).registerDefaultEvents()
        self.observable().addEvent(RunModelMixin.RUN_PHASE_CHANGED_EVENT)
        self.observable().addEvent(RunModelMixin.RUN_FAILED_EVENT)
        self.observable().addEvent(RunModelMixin.RUN_FINISHED_EVENT)


    def __init__(self, phase_count=1, *args):
        super(RunModelMixin, self).__init__(*args)
        self.__phase = 0
        self.__phase_count = phase_count
        self.__phase_update_count = 0

        self.__job_start_time  = 0
        self.__job_stop_time = 0

    def startSimulations(self):
        raise AbstractMethodError(self, "startSimulations")

    def killAllSimulations(self):
        raise AbstractMethodError(self, "killAllSimulations")

    def phaseCount(self):
        return self.__phase_count

    def currentPhase(self):
        return self.__phase

    def isFinished(self):
        return self.__phase == self.__phase_count

    def setPhase(self, phase):
        if not 0 <= phase <= self.__phase_count:
            raise ValueError("Phase must be an integer from 0 to less than %d." % self.__phase_count)

        if phase == 0:
            self.__job_start_time = int(time.time())

        if phase == self.__phase_count:
            self.__job_stop_time = int(time.time())

        self.__phase = phase
        self.__phase_update_count = 0
        self.observable().notify(RunModelMixin.RUN_PHASE_CHANGED_EVENT)


    def getRunningTime(self):
        if self.__job_stop_time < self.__job_start_time:
            return time.time() - self.__job_start_time
        else:
            return self.__job_stop_time - self.__job_start_time

    def runFailed(self, message=""):
        self.observable().notify(RunModelMixin.RUN_FAILED_EVENT, message)

    def getQueueSize(self):
        queue_size = len(self.ert().siteConfig().getJobQueue())

        if queue_size == 0:
            queue_size = 1

        return queue_size

    def getQueueStatus(self):
        """ @rtype: dict of (JobStatusType, int) """
        job_queue = self.ert().siteConfig().getJobQueue()

        queue_status = {}

        if job_queue.isRunning():
            for job_number in range(len(job_queue)):
                status = job_queue.getJobStatus(job_number)

                if not status in queue_status:
                    queue_status[status] = 0

                queue_status[status] += 1

        return queue_status

    def isQueueRunning(self):
        """ @rtype: bool """
        return self.ert().siteConfig().getJobQueue().isRunning()

    def getProgress(self):
        """ @rtype: float """
        if self.isFinished():
            current_progress = 1.0
        elif not self.isQueueRunning() and self.__phase_update_count > 0:
            current_progress = (self.__phase + 1.0) / self.__phase_count
        else:
            self.__phase_update_count += 1
            queue_status = self.getQueueStatus()
            queue_size = self.getQueueSize()

            done_state = JobStatusType.JOB_QUEUE_SUCCESS | JobStatusType.JOB_QUEUE_DONE
            done_count = 0

            for state in queue_status:
                if state in done_state:
                    done_count += queue_status[state]

            phase_progress = float(done_count) / queue_size
            current_progress = (self.__phase + phase_progress) / self.__phase_count

        return current_progress

    def ert(self):
        """ @rtype: EnkfMain """
        pass  # gets implemented by extending classes (ErtConnector) -> (todo) should make this as an abstract implementing class...



