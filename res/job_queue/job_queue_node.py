from cwrap import BaseCClass
from ecl.util.util import StringList
from res import ResPrototype
from res.job_queue import JobStatusType, ThreadStatus, JobSubmitStatusType
from threading import Thread, Lock

import time
import logging

logger = logging.getLogger(__name__)


class JobQueueNode(BaseCClass):
    TYPE_NAME = "job_queue_node"

    _alloc = ResPrototype(
        "void* job_queue_node_alloc_python(char*,"
        "char*,"
        "char*,"
        "int, "
        "stringlist,"
        "int, "
        "char*,"
        "char*,"
        "char*"
        ")",
        bind=False,
    )
    _free = ResPrototype("void job_queue_node_free(job_queue_node)")
    _submit = ResPrototype(
        "job_submit_status_type_enum job_queue_node_submit_simple(job_queue_node, driver)"
    )
    _kill = ResPrototype("bool job_queue_node_kill_simple(job_queue_node, driver)")

    _get_status = ResPrototype(
        "job_status_type_enum job_queue_node_get_status(job_queue_node)"
    )
    _update_status = ResPrototype(
        "bool job_queue_node_update_status_simple(job_queue_node, driver)"
    )
    _set_status = ResPrototype(
        "void job_queue_node_set_status(job_queue_node, job_status_type_enum)"
    )
    _get_submit_attempt = ResPrototype(
        "int job_queue_node_get_submit_attempt(job_queue_node)"
    )

    def __init__(
        self,
        job_script,
        job_name,
        run_path,
        num_cpu,
        status_file,
        ok_file,
        exit_file,
        done_callback_function,
        exit_callback_function,
        callback_arguments,
        max_runtime=None,
        callback_timeout=None,
    ):
        self.done_callback_function = done_callback_function
        self.exit_callback_function = exit_callback_function
        self.callback_timeout = callback_timeout
        self.callback_arguments = callback_arguments
        argc = 1
        argv = StringList()
        argv.append(run_path)

        self._thread_status = ThreadStatus.READY
        self._thread = None
        self._mutex = Lock()

        self.run_path = run_path
        self._max_runtime = max_runtime
        self._start_time = None
        self._end_time = None
        self._timed_out = False
        c_ptr = self._alloc(
            job_name,
            run_path,
            job_script,
            argc,
            argv,
            num_cpu,
            ok_file,
            status_file,
            exit_file,
            None,
            None,
            None,
            None,
        )

        if c_ptr is not None:
            super(JobQueueNode, self).__init__(c_ptr)
        else:
            raise ValueError("Unable to create job node object")

    def free(self):
        self._free()

    @property
    def timed_out(self):
        with self._mutex:
            return self._timed_out

    @property
    def submit_attempt(self):
        return self._get_submit_attempt()

    @property
    def status(self):
        return self._get_status()

    @property
    def thread_status(self):
        return self._thread_status

    def submit(self, driver):
        return self._submit(driver)

    def run_done_callback(self):
        callback_status = self.done_callback_function(self.callback_arguments)

        if callback_status:
            self._set_status(JobStatusType.JOB_QUEUE_SUCCESS)
        else:
            self._set_status(JobStatusType.JOB_QUEUE_EXIT)

        return callback_status

    def run_exit_callback(self):
        return self.exit_callback_function(self.callback_arguments)

    def is_running(self):
        return (
            self.status == JobStatusType.JOB_QUEUE_PENDING
            or self.status == JobStatusType.JOB_QUEUE_SUBMITTED
            or self.status == JobStatusType.JOB_QUEUE_RUNNING
            or self.status == JobStatusType.JOB_QUEUE_UNKNOWN
        )  # dont stop monitoring if LSF commands are unavailable

    @property
    def runtime(self):
        if self._start_time is None:
            return 0

        if self._end_time is None:
            return time.time() - self._start_time

        return self._end_time - self._start_time

    def _should_be_killed(self):
        return self.thread_status == ThreadStatus.STOPPING or (
            self._max_runtime is not None and self.runtime >= self._max_runtime
        )

    def _job_monitor(self, driver, pool_sema, max_submit):

        submit_status = self.submit(driver)
        if submit_status is not JobSubmitStatusType.SUBMIT_OK:
            self._set_status(JobStatusType.JOB_QUEUE_DONE)

        self.update_status(driver)

        while self.is_running():
            if (
                self._start_time is None
                and self.status == JobStatusType.JOB_QUEUE_RUNNING
            ):
                self._start_time = time.time()
            time.sleep(1)
            self.update_status(driver)
            if self._should_be_killed():
                self._kill(driver)
                if self._max_runtime and self.runtime >= self._max_runtime:
                    logger.error(f"MAX_RUNTIME reached in run path {self.run_path}")
                    if self.callback_timeout:
                        self.callback_timeout(self.callback_arguments)
                    with self._mutex:
                        self._timed_out = True

        self._end_time = time.time()

        with self._mutex:
            if self.status == JobStatusType.JOB_QUEUE_DONE:
                with pool_sema:
                    self.run_done_callback()

            if self.status == JobStatusType.JOB_QUEUE_SUCCESS:
                pass
            elif self.status == JobStatusType.JOB_QUEUE_EXIT:
                if self.submit_attempt < max_submit:
                    self._set_thread_status(ThreadStatus.READY)
                    return
                else:
                    self._set_status(JobStatusType.JOB_QUEUE_FAILED)
                    self.run_exit_callback()
            elif self.status == JobStatusType.JOB_QUEUE_IS_KILLED:
                pass
            else:
                self._set_thread_status(ThreadStatus.FAILED)
                raise AssertionError(
                    "Unexpected job status type after running job: {}".format(
                        self.status
                    )
                )

            self._set_thread_status(ThreadStatus.DONE)

    def run(self, driver, pool_sema, max_submit=2):
        # Prevent multiple threads working on the same object
        self.wait_for()
        # Do not start if already kill signal is sent
        if self.thread_status == ThreadStatus.STOPPING:
            self._set_thread_status(ThreadStatus.DONE)
            return

        self._set_thread_status(ThreadStatus.RUNNING)
        self._start_time = None
        self._thread = Thread(
            target=self._job_monitor, args=(driver, pool_sema, max_submit)
        )
        self._thread.start()

    def stop(self):
        with self._mutex:
            if self.thread_status == ThreadStatus.RUNNING:
                self._set_thread_status(ThreadStatus.STOPPING)
            elif self.thread_status == ThreadStatus.READY:
                # clean-up to get the correct status after being stopped by user
                self._set_thread_status(ThreadStatus.DONE)
                self._set_status(JobStatusType.JOB_QUEUE_FAILED)

            assert self.thread_status in [
                ThreadStatus.DONE,
                ThreadStatus.STOPPING,
                ThreadStatus.FAILED,
            ]

    def wait_for(self):
        if self._thread is not None and self._thread.is_alive():
            self._thread.join()

    def update_status(self, driver):
        if self.status != JobStatusType.JOB_QUEUE_WAITING:
            self._update_status(driver)

    def _set_thread_status(self, new_status):
        self._thread_status = new_status
