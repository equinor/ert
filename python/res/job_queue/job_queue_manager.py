#  Copyright (C) 2014  Equinor ASA, Norway.
#
#  The file 'job_queue_manager.py' is part of ERT - Ensemble based Reservoir Tool.
#
#  ERT is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  ERT is distributed in the hope that it will be useful, but WITHOUT ANY
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or
#  FITNESS FOR A PARTICULAR PURPOSE.
#
#  See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
#  for more details.
"""
Module implementing a queue for managing external jobs.

"""
from res.job_queue import Job, JobStatusType, ThreadStatus
from threading import BoundedSemaphore
import time

CONCURRENT_INTERNALIZATION = 10


class JobQueueManager:
    def __init__(self, queue, queue_evaluators=None):
        self._queue = queue
        self._queue_evaluators = queue_evaluators
        self._pool_sema = BoundedSemaphore(value=CONCURRENT_INTERNALIZATION)

    @property
    def queue(self):
        return self._queue

    def stop_queue(self):
        self.queue.kill_all_jobs()

    def getNumRunning(self):
        return self.queue.count_status(JobStatusType.JOB_QUEUE_RUNNING)

    def getNumWaiting(self):
        return self.queue.count_status(JobStatusType.JOB_QUEUE_WAITING)

    def getNumPending(self):
        return self.queue.count_status(JobStatusType.JOB_QUEUE_PENDING)

    def getNumSuccess(self):
        return self.queue.count_status(JobStatusType.JOB_QUEUE_SUCCESS)

    def getNumFailed(self):
        return self.queue.count_status(JobStatusType.JOB_QUEUE_FAILED)

    def isRunning(self):
        return self.queue.is_active()

    def isJobComplete(self, job_index):
        return not (
            self.queue.job_list[job_index].is_running()
            or self.queue.job_list[job_index].status == JobStatusType.JOB_QUEUE_WAITING
        )

    def isJobRunning(self, job_index):
        return self.queue.job_list[job_index].status == JobStatusType.JOB_QUEUE_RUNNING

    def isJobWaiting(self, job_index):
        return self.queue.job_list[job_index].status == JobStatusType.JOB_QUEUE_WAITING

    def didJobFail(self, job_index):
        return self.queue.job_list[job_index].status == JobStatusType.JOB_QUEUE_FAILED

    def didJobSucceed(self, job_index):
        return self.queue.job_list[job_index].status == JobStatusType.JOB_QUEUE_SUCCESS

    def getJobStatus(self, job_index):
        # See comment about return type in the prototype section at
        # the top of class.
        """ @rtype: res.job_queue.job_status_type_enum.JobStatusType """
        int_status = self.queue.job_list[job_index].status
        return JobStatusType(int_status)

    def __repr__(self):
        nw = self.getNumWaiting()
        nr = self.getNumRunning()
        ns = self.getNumSuccess()
        nf = self.getNumFailed()
        ir = "running" if self.isRunning() else "not running"
        status = "waiting=%d, running=%d, success=%d, failed=%d" % (nw, nr, ns, nf)
        return "JobQueueManager(%s, %s)" % (status, ir)

    def max_running(self):
        if self.queue.get_max_running() == 0:
            return len(self.queue.job_list)
        else:
            return self.queue.get_max_running()

    def _available_capacity(self):
        return (
            not self.queue.stopped and self.queue.count_running() < self.max_running()
        )

    def _launch_jobs(self):
        # Start waiting jobs
        while self._available_capacity():
            job = self.queue.fetch_next_waiting()
            if job is None:
                break
            job.run(
                driver=self.queue.driver,
                pool_sema=self._pool_sema,
                max_submit=self.queue.max_submit,
            )

    def _stop_jobs(self):
        for job in self.queue.job_list:
            job.stop()
        while self.queue.is_active():
            time.sleep(1)

    def _assert_complete(self):
        for job in self.queue.job_list:
            if job.thread_status != ThreadStatus.DONE:
                msg = "Unexpected job status type after running job: {} with thread status: {}"
                raise AssertionError(msg.format(job.status, job.thread_status))

    def execute_queue(self):
        while self.queue.is_active() and not self.queue.stopped:
            self._launch_jobs()

            time.sleep(1)

            if self._queue_evaluators is not None:
                for func in self._queue_evaluators:
                    func()

        if self.queue.stopped:
            self._stop_jobs()

        self._assert_complete()
