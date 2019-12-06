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
from cwrap import BaseCClass
from res import ResPrototype
from res.job_queue import Job, JobStatusType, ThreadStatus
from threading import BoundedSemaphore
import time

CONCURRENT_INTERNALIZATION = 10

class JobQueueManager(BaseCClass):
    TYPE_NAME = "job_queue_manager"
    _alloc           = ResPrototype("void* job_queue_manager_alloc( job_queue)", bind = False)
    _free            = ResPrototype("void job_queue_manager_free( job_queue_manager )")
    _start_queue     = ResPrototype("void job_queue_manager_start_queue( job_queue_manager , int , bool)")
    _stop_queue      = ResPrototype("void job_queue_manager_stop_queue(job_queue_manager)")
    _get_num_waiting = ResPrototype("int job_queue_manager_get_num_waiting( job_queue_manager )")
    _get_num_pending = ResPrototype("int job_queue_manager_get_num_pending( job_queue_manager )")
    _get_num_running = ResPrototype("int job_queue_manager_get_num_running( job_queue_manager )")
    _get_num_success = ResPrototype("int job_queue_manager_get_num_success( job_queue_manager )")
    _get_num_failed  = ResPrototype("int job_queue_manager_get_num_failed( job_queue_manager )")
    _is_running      = ResPrototype("bool job_queue_manager_is_running( job_queue_manager )")
    _job_complete    = ResPrototype("bool job_queue_manager_job_complete( job_queue_manager , int)")
    _job_running     = ResPrototype("bool job_queue_manager_job_running( job_queue_manager , int)")

    # Note, even if all realizations have finished, they need not all be failed or successes.
    # That is how Ert report things. They can be "killed", which is neither success nor failure.
    _job_failed      = ResPrototype("bool job_queue_manager_job_failed( job_queue_manager , int)")
    _job_waiting     = ResPrototype("bool job_queue_manager_job_waiting( job_queue_manager , int)")
    _job_success     = ResPrototype("bool job_queue_manager_job_success( job_queue_manager , int)")

    # The return type of the job_queue_manager_iget_job_status should
    # really be the enum job_status_type_enum, but I just did not
    # manage to get the prototyping right. Have therefor taken the
    # return as an integer and convert it in the getJobStatus()
    # method.
    _job_status      = ResPrototype("int job_queue_manager_iget_job_status(job_queue_manager, int)")

    def __init__(self, queue, queue_evaluators=None):
        c_ptr = self._alloc(queue)
        self._queue = queue
        self._queue_evaluators = queue_evaluators
        self._pool_sema = BoundedSemaphore(value=CONCURRENT_INTERNALIZATION)
        super(JobQueueManager, self).__init__(c_ptr)

    @property
    def queue(self):
        return self._queue

    def stop_queue(self):
        self.queue.kill_all_jobs()

    def startQueue(self , total_size , verbose = False ):
        self._start_queue( total_size , verbose )

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

    def free(self):
        self._free( )

    def isJobComplete(self, job_index):
        return not (self.queue.job_list[job_index].is_running()
                    or self.queue.job_list[job_index].status == JobStatusType.JOB_QUEUE_WAITING)

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
        nw = self._get_num_waiting()
        nr = self._get_num_running()
        ns = self._get_num_success()
        nf = self._get_num_failed()
        ir = 'running' if self._is_running() else 'not running'
        return 'JobQueueManager(waiting=%d, running=%d, success=%d, failed=%d, %s)' % (nw,nr,ns,nf,ir)

    def max_running(self):
        if self.queue.get_max_running() == 0:
            return len(self.queue.job_list)
        else:
            return self.queue.get_max_running()

    def _available_capacity(self):
        return not self.queue.stopped and self.queue.count_running() < self.max_running()

    def _launch_jobs(self):
        #Start waiting jobs
        while self._available_capacity():
            job = self.queue.fetch_next_waiting()
            if job is None:
                break
            job.run(
                driver=self.queue.driver,
                pool_sema=self._pool_sema,
                max_submit=self.queue.max_submit
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
