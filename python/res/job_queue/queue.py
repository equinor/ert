#  Copyright (C) 2011  Equinor ASA, Norway.
#
#  The file 'job_queue.py' is part of ERT - Ensemble based Reservoir Tool.
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

import asyncio
import sys
import time
import ctypes

from cwrap import BaseCClass

from res import ResPrototype
from res.job_queue import Job, JobStatusType, ThreadStatus, JobQueueNode

LONG_RUNNING_FACTOR = 1.25


class JobQueue(BaseCClass):
    # If the queue is created with size == 0 that means that it will
    # just grow as needed; for the queue layer to know when to exit
    # you must call the function submit_complete() when you have no
    # more jobs to submit.
    #
    # If the number of jobs is known in advance you can create the
    # queue with a finite value for size, in that case it is not
    # necessary to explitly inform the queue layer when all jobs have
    # been submitted.
    TYPE_NAME = "job_queue"
    _alloc = ResPrototype(
        "void* job_queue_alloc( int , char* , char* , char* )", bind=False
    )
    _start_user_exit = ResPrototype("bool job_queue_start_user_exit( job_queue )")
    _get_user_exit = ResPrototype("bool job_queue_get_user_exit( job_queue )")
    _free = ResPrototype("void job_queue_free( job_queue )")
    _set_max_running = ResPrototype("void job_queue_set_max_running( job_queue , int)")
    _get_max_running = ResPrototype("int  job_queue_get_max_running( job_queue )")
    _set_max_job_duration = ResPrototype(
        "void job_queue_set_max_job_duration( job_queue , int)"
    )
    _get_max_job_duration = ResPrototype(
        "int  job_queue_get_max_job_duration( job_queue )"
    )
    _set_driver = ResPrototype("void job_queue_set_driver( job_queue , void* )")
    _kill_job = ResPrototype("bool job_queue_kill_job( job_queue , int )")
    _start_queue = ResPrototype("void job_queue_run_jobs( job_queue , int , bool)")
    _run_jobs = ResPrototype("void job_queue_run_jobs_threaded(job_queue , int , bool)")
    _sim_start = ResPrototype("time_t job_queue_iget_sim_start( job_queue , int)")
    _iget_driver_data = ResPrototype(
        "void* job_queue_iget_driver_data( job_queue , int)"
    )

    _num_running = ResPrototype("int  job_queue_get_num_running( job_queue )")
    _num_complete = ResPrototype("int  job_queue_get_num_complete( job_queue )")
    _num_waiting = ResPrototype("int  job_queue_get_num_waiting( job_queue )")
    _num_pending = ResPrototype("int  job_queue_get_num_pending( job_queue )")

    _is_running = ResPrototype("bool job_queue_is_running( job_queue )")
    _submit_complete = ResPrototype("void job_queue_submit_complete( job_queue )")
    _iget_sim_start = ResPrototype("time_t job_queue_iget_sim_start( job_queue , int)")
    _get_active_size = ResPrototype("int  job_queue_get_active_size( job_queue )")
    _get_pause = ResPrototype("bool job_queue_get_pause(job_queue)")
    _set_pause_on = ResPrototype("void job_queue_set_pause_on(job_queue)")
    _set_pause_off = ResPrototype("void job_queue_set_pause_off(job_queue)")
    _get_max_submit = ResPrototype("int job_queue_get_max_submit(job_queue)")

    _get_job_status = ResPrototype(
        "job_status_type_enum job_queue_iget_job_status(job_queue, int)"
    )

    _get_ok_file = ResPrototype("char* job_queue_get_ok_file(job_queue)")
    _get_exit_file = ResPrototype("char* job_queue_get_exit_file(job_queue)")
    _get_status_file = ResPrototype("char* job_queue_get_status_file(job_queue)")
    _add_job = ResPrototype("int job_queue_add_job_node(job_queue, job_queue_node)")

    def __repr__(self):
        nrun, ncom, nwait, npend = (
            self._num_running(),
            self._num_complete(),
            self._num_waiting(),
            self._num_pending(),
        )
        isrun = "running" if self.isRunning() else "not running"
        cnt = "%s, num_running=%d, num_complete=%d, num_waiting=%d, num_pending=%d, active=%d"
        return self._create_repr(cnt % (isrun, nrun, ncom, nwait, npend, len(self)))

    def __init__(self, driver, max_submit=2, size=0):
        """
        Short doc...
        The @max_submit argument says how many times the job be submitted (including a failure)
              max_submit = 2: means that we can submit job once more
        The @size argument is used to say how many jobs the queue will
        run, in total.
              size = 0: That means that you do not tell the queue in
                advance how many jobs you have. The queue will just run
                all the jobs you add, but you have to inform the queue in
                some way that all jobs have been submitted. To achieve
                this you should call the submit_complete() method when all
                jobs have been submitted.#

              size > 0: The queue will know exactly how many jobs to run,
                and will continue until this number of jobs have completed
                - it is not necessary to call the submit_complete() method
                in this case.
        """

        OK_file = "OK"
        status_file = "STATUS"
        exit_file = "ERROR"
        self.job_list = []
        self._stopped = False
        c_ptr = self._alloc(max_submit, OK_file, status_file, exit_file)
        super(JobQueue, self).__init__(c_ptr)
        self.size = size

        self.driver = driver
        self._set_driver(driver.from_param(driver))

    def kill_job(self, queue_index):
        """
        Will kill job nr @index.
        """
        self._kill_job(queue_index)

    def start(self, blocking=False):
        verbose = False
        self._run_jobs(self.size, verbose)

    def clear(self):
        pass

    def block_waiting(self):
        """
        Will block as long as there are waiting jobs.
        """
        while self.num_waiting > 0:
            time.sleep(1)

    def block(self):
        """
        Will block as long as there are running jobs.
        """
        while self.isRunning:
            time.sleep(1)

    def submit_complete(self):
        """
        Method to inform the queue that all jobs have been submitted.

        If the queue has been created with size == 0 the queue has no
        way of knowing when all jobs have completed; hence in that
        case you must call the submit_complete() method when all jobs
        have been submitted.

        If you know in advance exactly how many jobs you will run that
        should be specified with the size argument when creating the
        queue, in that case it is not necessary to call the
        submit_complete() method.
        """
        self._submit_complete()

    def isRunning(self):
        return self._is_running()

    def num_running(self):
        return self._num_running()

    def num_pending(self):
        return self._num_pending()

    def num_waiting(self):
        return self._num_waiting()

    def num_complete(self):
        return self._num_complete()

    def __getitem__(self, index):
        idx = index
        ls = len(self)
        if idx < 0:
            idx += ls
        if 0 <= idx < ls:
            return self._iget_driver_data(idx)
        raise IndexError(
            "index out of range, was: %d should be in [0, %d)" % (index, ls)
        )

    def exists(self, index):
        return self[index]

    def get_max_running(self):
        return self.driver.get_max_running()

    def set_max_running(self, max_running):
        self.driver.set_max_running(max_running)

    def get_max_job_duration(self):
        return self._get_max_job_duration()

    def set_max_job_duration(self, max_duration):
        self._set_max_job_duration(max_duration)

    @property
    def max_submit(self):
        return self._get_max_submit()

    def killAllJobs(self):
        # The queue will not set the user_exit flag before the
        # queue is in a running state. If the queue does not
        # change to running state within a timeout the C function
        # will return False, and that False value is just passed
        # along.
        user_exit = self._start_user_exit()
        if user_exit:
            while self.isRunning():
                time.sleep(0.1)
            return True
        else:
            return False

    def igetSimStart(self, job_index):
        return self._iget_sim_start(self, job_index)

    def getUserExit(self):
        # Will check if a user_exit has been initated on the job. The
        # queue can be queried about this status until a
        # job_queue_reset() call is invoked, and that should not be
        # done before the queue is recycled to run another batch of
        # simulations.
        return self._get_user_exit()

    def set_pause_on(self):
        self._set_pause_on()

    def set_pause_off(self):
        self._set_pause_off()

    def free(self):
        self._free()

    def __len__(self):
        return self._get_active_size()

    def getJobStatus(self, job_number):
        """ @rtype: JobStatusType """
        return self._get_job_status(job_number)

    def is_active(self):
        for job in self.job_list:
            if (
                job.thread_status == ThreadStatus.READY
                or job.thread_status == ThreadStatus.RUNNING
                or job.thread_status == ThreadStatus.STOPPING
            ):
                return True
        return False

    def fetch_next_waiting(self):
        for job in self.job_list:
            if job.thread_status == ThreadStatus.READY:
                return job
        return None

    def count_status(self, status):
        return len([job for job in self.job_list if job.status == status])

    @property
    def stopped(self):
        return self._stopped

    def kill_all_jobs(self):
        self._stopped = True

    @property
    def queue_size(self):
        return len(self.job_list)

    @property
    def ok_file(self):
        return self._get_ok_file()

    @property
    def exit_file(self):
        return self._get_exit_file()

    @property
    def status_file(self):
        return self._get_status_file()

    def add_job(self, job):
        job.convertToCReference(None)
        queue_index = self._add_job(job)
        self.job_list.append(job)
        return queue_index

    def count_running(self):
        return sum(job.thread_status == ThreadStatus.RUNNING for job in self.job_list)

    def max_running(self):
        if self.get_max_running() == 0:
            return len(self.job_list)
        else:
            return self.get_max_running()

    def available_capacity(self):
        return not self.stopped and self.count_running() < self.max_running()

    def stop_jobs(self):
        for job in self.job_list:
            job.stop()
        while self.is_active():
            time.sleep(1)

    async def stop_jobs_async(self):
        for job in self.job_list:
            job.stop()
        while self.is_active():
            await asyncio.sleep(1)

    def assert_complete(self):
        for job in self.job_list:
            if job.thread_status != ThreadStatus.DONE:
                msg = "Unexpected job status type after running job: {} with thread status: {}"
                raise AssertionError(msg.format(job.status, job.thread_status))

    def launch_jobs(self, pool_sema):
        # Start waiting jobs
        while self.available_capacity():
            job = self.fetch_next_waiting()
            if job is None:
                break
            job.run(
                driver=self.driver,
                pool_sema=pool_sema,
                max_submit=self.max_submit,
            )

    def execute_queue(self, pool_sema, evaluators):
        while self.is_active() and not self.stopped:
            self.launch_jobs(pool_sema)

            time.sleep(1)

            if evaluators is not None:
                for func in evaluators:
                    func()

        if self.stopped:
            self.stop_jobs()

        self.assert_complete()

    def add_job_from_run_arg(self, run_arg, res_config, max_runtime, ok_cb, exit_cb):
        job_name = run_arg.job_name
        run_path = run_arg.runpath
        job_script = res_config.queue_config.job_script
        num_cpu = res_config.queue_config.num_cpu
        if num_cpu == 0:
            num_cpu = res_config.ecl_config.num_cpu

        job = JobQueueNode(
            job_script=job_script,
            job_name=job_name,
            run_path=run_path,
            num_cpu=num_cpu,
            status_file=self.status_file,
            ok_file=self.ok_file,
            exit_file=self.exit_file,
            done_callback_function=ok_cb,
            exit_callback_function=exit_cb,
            callback_arguments=[run_arg, res_config],
            max_runtime=max_runtime,
        )

        if job is None:
            return
        run_arg._set_queue_index(self.add_job(job))

    def add_ee_stage(self, stage):
        job = JobQueueNode(
            job_script=stage.get_job_script(),
            job_name=stage.get_job_name(),
            run_path=stage.get_run_path(),
            num_cpu=stage.get_num_cpu(),
            status_file=self.status_file,
            ok_file=self.ok_file,
            exit_file=self.exit_file,
            done_callback_function=stage.get_done_callback(),
            exit_callback_function=stage.get_exit_callback(),
            callback_arguments=stage.get_callback_arguments(),
            max_runtime=stage.get_max_runtime(),
        )
        if job is None:
            raise ValueError("JobQueueNode constructor created None job")

        stage.get_run_arg()._set_queue_index(self.add_job(job))

    def stop_long_running_jobs(self, minimum_required_realizations):
        finished_realizations = self.count_status(JobStatusType.JOB_QUEUE_DONE)
        if finished_realizations < minimum_required_realizations:
            return

        completed_jobs = [
            job for job in self.job_list if job.status == JobStatusType.JOB_QUEUE_DONE
        ]
        average_runtime = sum([job.runtime for job in completed_jobs]) / float(
            len(completed_jobs)
        )

        for job in self.job_list:
            if job.runtime > LONG_RUNNING_FACTOR * average_runtime:
                job.stop()
