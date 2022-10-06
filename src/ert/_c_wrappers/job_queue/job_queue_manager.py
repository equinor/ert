"""
Module implementing a queue for managing external jobs.

"""
from threading import BoundedSemaphore
from typing import TYPE_CHECKING, Any

from ert._c_wrappers.job_queue.job_status_type_enum import JobStatusType

if TYPE_CHECKING:
    from ert._c_wrappers.job_queue import JobQueue


CONCURRENT_INTERNALIZATION = 1


# TODO: there's no need for this class, all the behavior belongs in the queue
# class proper.
class JobQueueManager:
    def __init__(self, queue: "JobQueue", queue_evaluators: Any = None):
        self._queue = queue
        self._queue_evaluators = queue_evaluators
        self._pool_sema = BoundedSemaphore(value=CONCURRENT_INTERNALIZATION)

    @property
    def queue(self) -> "JobQueue":
        return self._queue

    def stop_queue(self) -> None:
        self.queue.kill_all_jobs()

    def getNumRunning(self) -> int:
        return self.queue.count_status(JobStatusType.JOB_QUEUE_RUNNING)

    def getNumWaiting(self) -> int:
        return self.queue.count_status(JobStatusType.JOB_QUEUE_WAITING)

    def getNumPending(self) -> int:
        return self.queue.count_status(JobStatusType.JOB_QUEUE_PENDING)

    def getNumSuccess(self) -> int:
        return self.queue.count_status(JobStatusType.JOB_QUEUE_SUCCESS)

    def getNumFailed(self) -> int:
        return self.queue.count_status(JobStatusType.JOB_QUEUE_FAILED)

    def isRunning(self) -> bool:
        return self.queue.is_active()

    def isJobComplete(self, job_index: int) -> bool:
        return not (
            self.queue.job_list[job_index].is_running()
            or self.queue.job_list[job_index].status == JobStatusType.JOB_QUEUE_WAITING
        )

    def isJobRunning(self, job_index: int) -> bool:
        return self.queue.job_list[job_index].status == JobStatusType.JOB_QUEUE_RUNNING

    def isJobWaiting(self, job_index: int) -> bool:
        return self.queue.job_list[job_index].status == JobStatusType.JOB_QUEUE_WAITING

    def didJobFail(self, job_index: int) -> bool:
        return self.queue.job_list[job_index].status == JobStatusType.JOB_QUEUE_FAILED

    def didJobSucceed(self, job_index: int) -> bool:
        return self.queue.job_list[job_index].status == JobStatusType.JOB_QUEUE_SUCCESS

    def getJobStatus(self, job_index: int) -> JobStatusType:
        # See comment about return type in the prototype section at
        # the top of class.
        int_status = self.queue.job_list[job_index].status
        return JobStatusType(int_status)

    def __repr__(self) -> str:
        return (
            "JobQueueManager("
            f"waiting={self.getNumWaiting()}, running={self.getNumRunning()}, "
            f"success={self.getNumSuccess()}, failed={self.getNumFailed()}"
            f", {'running' if self.isRunning() else 'not running'}"
            ")"
        )

    def execute_queue(self):
        self._queue.execute_queue(self._pool_sema, self._queue_evaluators)
