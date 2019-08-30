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
from res.job_queue import Job, JobStatusType
import time

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
    _status_timestamp= ResPrototype("time_t job_queue_manager_get_status_timestamp(job_queue_manager)")
    _global_progress_timestamp = ResPrototype("time_t job_queue_manager_get_progress_timestamp(job_queue_manager)")
    _iget_progress_timestamp = ResPrototype("time_t job_queue_manager_iget_progress_timestamp(job_queue_manager, int)")

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

    def __init__(self, queue):
        c_ptr = self._alloc(queue)
        self.queue = queue
        super(JobQueueManager, self).__init__(c_ptr)

    def status_timestamp(self):
        """
        Will return a datetime object corresponding to the last status change
        """
        ts = self._status_timestamp()
        return ts.datetime()

    def progress_timestamp(self, job_index = None):
        """Will return the timestamp of last progress update.

        By default the timestamp will be global, i.e. for all jobs, but if you
        pass a job_index as optional argument the timestamp will correspond to
        that job.
        """
        if job_index is None:
            ts = self._global_progress_timestamp()
        else:
            ts = self._iget_progress_timestamp(job_index)

        return ts.datetime()


    def get_job_queue(self):
        return self.queue

    def stop_queue(self):
        self._stop_queue( )

    def startQueue(self , total_size , verbose = False ):
        self._start_queue( total_size , verbose )

    def getNumRunning(self):
        return self._get_num_running(  )

    def getNumWaiting(self):
        return self._get_num_waiting( )

    def getNumPending(self):
        return self._get_num_pending( )


    def getNumSuccess(self):
        return self._get_num_success( )

    def getNumFailed(self):
        return self._get_num_failed(  )

    def isRunning(self):
        return self._is_running( )

    def free(self):
        self._free( )

    def isJobComplete(self, job_index):
        return self._job_complete( job_index )

    def isJobRunning(self, job_index):
        return self._job_running( job_index )

    def isJobWaiting(self, job_index):
        return self._job_waiting( job_index )

    def didJobFail(self, job_index):
        return self._job_failed( job_index )

    def didJobSucceed(self, job_index):
        return self._job_success( job_index )

    def getJobStatus(self, job_index):
        # See comment about return type in the prototype section at
        # the top of class.
        """ @rtype: res.job_queue.job_status_type_enum.JobStatusType """
        int_status = self._job_status(job_index)
        return JobStatusType( int_status )


    def __repr__(self):
        nw = self._get_num_waiting()
        nr = self._get_num_running()
        ns = self._get_num_success()
        nf = self._get_num_failed()
        ir = 'running' if self._is_running() else 'not running'
        return 'JobQueueManager(waiting=%d, running=%d, success=%d, failed=%d, %s)' % (nw,nr,ns,nf,ir)

    def max_running(self):
        return self.get_job_queue().get_max_running()

    def execute_queue(self):

        job_queue = self.get_job_queue()
        started_job_threads = []
        while job_queue.is_running():
            job = job_queue.fetch_next_waiting()
            while job is not None and job_queue.count_running() <= self.max_running():
                started_job_threads.append(job.run(job_queue.driver))
                job = job_queue.fetch_next_waiting()
            time.sleep(1)

        for thread in started_job_threads:
            thread.join()
