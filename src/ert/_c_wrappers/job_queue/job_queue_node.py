import logging
import time
from threading import Lock, Thread

from cwrap import BaseCClass
from ecl.util.util import StringList

from ert._c_wrappers import ResPrototype
from ert._c_wrappers.job_queue.job_status_type_enum import JobStatusType
from ert._c_wrappers.job_queue.job_submit_status_type_enum import JobSubmitStatusType
from ert._c_wrappers.job_queue.thread_status_type_enum import ThreadStatus
from ert._clib.model_callbacks import LoadStatus

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
        "job_submit_status_type_enum job_queue_node_submit_simple(job_queue_node, driver)"  # noqa
    )
    _run_kill = ResPrototype("bool job_queue_node_kill_simple(job_queue_node, driver)")

    _get_status = ResPrototype(
        "job_status_type_enum job_queue_node_get_status(job_queue_node)"
    )

    _refresh_status = ResPrototype(
        "job_status_type_enum job_queue_node_refresh_status(job_queue_node, driver)"
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
        self._tried_killing = 0

        self.run_path = run_path
        self._max_runtime = max_runtime
        self._start_time = None
        self._end_time = None
        self._timed_out = False
        self._status_msg = ""
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
            super().__init__(c_ptr)
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

    def refresh_status(self, driver):
        return self._refresh_status(driver)

    @property
    def status(self):
        return self._get_status()

    @property
    def thread_status(self):
        return self._thread_status

    def submit(self, driver):
        return self._submit(driver)

    def run_done_callback(self):
        callback_status, status_msg = self.done_callback_function(
            self.callback_arguments
        )
        if callback_status == LoadStatus.LOAD_SUCCESSFUL:
            self._set_status(JobStatusType.JOB_QUEUE_SUCCESS)
        elif callback_status == LoadStatus.TIME_MAP_FAILURE:
            self._set_status(JobStatusType.JOB_QUEUE_FAILED)
        else:
            self._set_status(JobStatusType.JOB_QUEUE_EXIT)
        self._status_msg = status_msg
        return callback_status

    def run_exit_callback(self):
        return self.exit_callback_function(self.callback_arguments)

    def is_running(self, given_status=None):
        status = given_status or self.status
        return status in (
            JobStatusType.JOB_QUEUE_PENDING,
            JobStatusType.JOB_QUEUE_SUBMITTED,
            JobStatusType.JOB_QUEUE_RUNNING,
            JobStatusType.JOB_QUEUE_UNKNOWN,
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

        current_status = self.refresh_status(driver)

        while self.is_running(current_status):
            if (
                self._start_time is None
                and current_status == JobStatusType.JOB_QUEUE_RUNNING
            ):
                self._start_time = time.time()
            time.sleep(1)
            if self._should_be_killed():
                self._kill(driver)
                if self._max_runtime and self.runtime >= self._max_runtime:
                    # We sometimes end up in a state where we are not able to kill it,
                    # so we end up flooding the logs with identical statements, so we
                    # check before we log.
                    if self._tried_killing == 1:
                        logger.info(f"MAX_RUNTIME reached in run path {self.run_path}")
                    elif self._tried_killing % 100 == 0:
                        logger.warning(
                            f"Tried killing with MAX_RUNTIME {self._tried_killing} "
                            f"times without success in {self.run_path}"
                        )
                    if self.callback_timeout:
                        self.callback_timeout(self.callback_arguments)
                    with self._mutex:
                        self._timed_out = True

            current_status = self.refresh_status(driver)

        self._end_time = time.time()

        with self._mutex:
            if current_status == JobStatusType.JOB_QUEUE_DONE:
                with pool_sema:
                    logger.info(
                        f"Realization: {self.callback_arguments[0].iens} complete, "
                        "starting to load results"
                    )
                    self.run_done_callback()

            # refresh cached status after running the callback
            current_status = self.refresh_status(driver)
            if current_status == JobStatusType.JOB_QUEUE_SUCCESS:
                pass
            elif current_status == JobStatusType.JOB_QUEUE_EXIT:
                if self.submit_attempt < max_submit:
                    logger.warning(
                        f"Realization: {self.callback_arguments[0].iens} "
                        f"failed with: {self._status_msg}, resubmitting"
                    )
                    self._set_thread_status(ThreadStatus.READY)
                    return
                else:
                    logger.error(
                        f"Realization: {self.callback_arguments[0].iens} "
                        f"failed after reaching max submit with: {self._status_msg}"
                    )
                    self._set_status(JobStatusType.JOB_QUEUE_FAILED)
                    self.run_exit_callback()
            elif current_status == JobStatusType.JOB_QUEUE_IS_KILLED:
                pass
            elif current_status == JobStatusType.JOB_QUEUE_FAILED:
                logger.error(
                    f"Realization: {self.callback_arguments[0].iens} "
                    f"failed with: {self._status_msg}"
                )
                self.run_exit_callback()
            else:
                self._set_thread_status(ThreadStatus.FAILED)
                raise AssertionError(
                    f"Unexpected job status type after "
                    f"running job: {current_status}"
                )

            self._set_thread_status(ThreadStatus.DONE)

    def _kill(self, driver):
        self._run_kill(driver)
        self._tried_killing += 1

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

    def _set_thread_status(self, new_status):
        self._thread_status = new_status
