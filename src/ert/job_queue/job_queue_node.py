from __future__ import annotations

import logging
import multiprocessing
import random
import time
from threading import Lock, Semaphore, Thread
from typing import TYPE_CHECKING, Optional

from cwrap import BaseCClass
from ecl.util.util import StringList

from ert._c_wrappers import ResPrototype
from ert.load_status import LoadStatus

from .job_status_type_enum import JobStatusType
from .job_submit_status_type_enum import JobSubmitStatusType
from .thread_status_type_enum import ThreadStatus

if TYPE_CHECKING:
    from ert.callbacks import Callback, CallbackArgs

    from .driver import Driver

logger = logging.getLogger(__name__)


# pylint: disable=too-many-instance-attributes
class JobQueueNode(BaseCClass):  # type: ignore
    TYPE_NAME = "job_queue_node"

    _alloc = ResPrototype(
        "void* job_queue_node_alloc_python(char*,"
        "char*,"
        "char*,"
        "int, "
        "stringlist,"
        "int, "
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

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        job_script: str,
        job_name: str,
        run_path: str,
        num_cpu: int,
        status_file: str,
        exit_file: str,
        done_callback_function: Callback,
        exit_callback_function: Callback,
        callback_arguments: CallbackArgs,
        max_runtime: Optional[int] = None,
        callback_timeout: Optional[Callback] = None,
    ):
        self.done_callback_function = done_callback_function
        self.exit_callback_function = exit_callback_function
        self.callback_timeout = callback_timeout
        self.callback_arguments = callback_arguments
        argc = 1
        argv = StringList()
        argv.append(run_path)

        self._thread_status: ThreadStatus = ThreadStatus.READY
        self._thread: Optional[Thread] = None
        self._mutex = Lock()
        self._tried_killing = 0

        self.run_path = run_path
        self._max_runtime = max_runtime
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._timed_out = False
        self._status_msg = ""
        self._job_name = job_name
        c_ptr = self._alloc(
            job_name,
            run_path,
            job_script,
            argc,
            argv,
            num_cpu,
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

    def free(self) -> None:
        self._free()

    def __str__(self) -> str:
        return (
            f"JobNode: Name:{self._job_name}, Status: {self.status}, "
            f"Timed_out: {self.timed_out}, "
            f"Submit_attempt: {self.submit_attempt}"
        )

    @property
    def timed_out(self) -> bool:
        with self._mutex:
            return self._timed_out

    @property
    def submit_attempt(self) -> int:
        return self._get_submit_attempt()  # type: ignore

    def refresh_status(self, driver: "Driver") -> JobStatusType:
        return self._refresh_status(driver)  # type: ignore

    @property
    def status(self) -> JobStatusType:
        return self._get_status()  # type: ignore

    @property
    def thread_status(self) -> ThreadStatus:
        return self._thread_status

    def submit(self, driver: "Driver") -> JobSubmitStatusType:
        return self._submit(driver)  # type: ignore

    def run_done_callback(self) -> Optional[LoadStatus]:
        # status_msg has a maximum length of 1024 bytes.
        # the size is immutable after creation due to being backed by a c array.
        status_msg = multiprocessing.Array("c", b" " * 1024)
        callback_status = multiprocessing.Value("i", 2)
        pcontext = multiprocessing.Process(
            target=self.done_callback_wrapper,
            args=self.callback_arguments,
            kwargs={
                "callback_status_shared": callback_status,
                "status_msg_shared": status_msg,
            },
        )
        pcontext.start()
        pcontext.join()

        # import moved here due to circular dependency error
        # pylint: disable=import-outside-toplevel
        from ert._c_wrappers.enkf import RealizationStateEnum

        # this step was added because the state_map update in
        #      forward_model_ok does not propagate from the spawned process.
        run_arg = self.callback_arguments[0]
        run_arg.ensemble_storage.state_map[run_arg.iens] = (
            RealizationStateEnum.STATE_HAS_DATA
            if callback_status.value == LoadStatus.LOAD_SUCCESSFUL.value
            else RealizationStateEnum.STATE_LOAD_FAILURE
        )

        if callback_status.value == LoadStatus.LOAD_SUCCESSFUL.value:
            self._set_status(JobStatusType.JOB_QUEUE_SUCCESS)
        elif callback_status.value == LoadStatus.TIME_MAP_FAILURE.value:
            self._set_status(JobStatusType.JOB_QUEUE_FAILED)
        else:
            self._set_status(JobStatusType.JOB_QUEUE_EXIT)
        self._status_msg = status_msg.value.decode("utf-8")
        return callback_status

    def done_callback_wrapper(
        self,
        *args,
        callback_status_shared: multiprocessing.Value,
        status_msg_shared: multiprocessing.Array,
    ) -> None:
        callback_status, status_msg = self.done_callback_function(*args)
        callback_status_shared.value = callback_status.value
        status_msg_shared.value = bytes(status_msg, "utf-8")

    def run_exit_callback(self) -> None:
        self.exit_callback_function(*self.callback_arguments)

    def is_running(self, given_status: Optional[JobStatusType] = None) -> bool:
        status = given_status or self.status
        return status in (
            JobStatusType.JOB_QUEUE_PENDING,
            JobStatusType.JOB_QUEUE_SUBMITTED,
            JobStatusType.JOB_QUEUE_RUNNING,
            JobStatusType.JOB_QUEUE_UNKNOWN,
        )  # dont stop monitoring if LSF commands are unavailable

    @property
    def runtime(self) -> float:
        if self._start_time is None:
            return 0

        if self._end_time is None:
            return time.time() - self._start_time

        return self._end_time - self._start_time

    # pylint: disable=too-many-branches, too-many-statements
    def _job_monitor(
        self, driver: "Driver", pool_sema: Semaphore, max_submit: int
    ) -> None:
        submit_status = self.submit(driver)
        if submit_status is not JobSubmitStatusType.SUBMIT_OK:
            self._set_status(JobStatusType.JOB_QUEUE_DONE)

        current_status = self.refresh_status(driver)

        # in the following loop, we increase the sleep time between loop iterations as
        # long running realizations do not change state often, and too frequent querying
        # with many realizations starves other threads for resources.
        initial_sleep_seconds = 1
        time_sleep_seconds = initial_sleep_seconds
        max_sleep_seconds = 30
        time_until_longer_sleep_seconds = 30
        use_random_sleep_offset = False
        while self.is_running(current_status):
            if (
                self._start_time is None
                and current_status == JobStatusType.JOB_QUEUE_RUNNING
            ):
                self._start_time = time.time()
            if (
                self._start_time is not None
                and time_sleep_seconds != max_sleep_seconds
                and time.time() - self._start_time > time_until_longer_sleep_seconds
            ):
                time_sleep_seconds = max_sleep_seconds
                use_random_sleep_offset = True
            time.sleep(
                time_sleep_seconds + use_random_sleep_offset * random.randint(-5, 5)
            )
            if self._max_runtime is not None and self.runtime >= self._max_runtime:
                self._kill(driver)
                # We sometimes end up in a state where we are not able to kill it,
                # so we end up flooding the logs with identical statements, so we
                # check before we log.
                if self._tried_killing == 1:
                    logger.error(
                        f"MAX_RUNTIME reached in run path {self.run_path}. Runtime: "
                        f"{self.runtime} (max runtime: {self._max_runtime})"
                    )
                elif self._tried_killing % 100 == 0:
                    logger.warning(
                        f"Tried killing with MAX_RUNTIME {self._tried_killing} "
                        f"times without success in {self.run_path}"
                    )
                if self.callback_timeout:
                    self.callback_timeout(*self.callback_arguments)
                with self._mutex:
                    self._timed_out = True

            elif self.thread_status == ThreadStatus.STOPPING:
                if self._tried_killing == 1:
                    logger.error(
                        f"Killing job in {self.run_path} ({self.thread_status})."
                    )
                self._kill(driver)

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
            elif current_status in [
                JobStatusType.JOB_QUEUE_IS_KILLED,
                JobStatusType.JOB_QUEUE_DO_KILL_NODE_FAILURE,
            ]:
                self.run_exit_callback()
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

    def _kill(self, driver: "Driver") -> None:
        self._run_kill(driver)
        self._tried_killing += 1

    def run(self, driver: "Driver", pool_sema: Semaphore, max_submit: int = 2) -> None:
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

    def stop(self) -> None:
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

    def wait_for(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            self._thread.join()

    def _set_thread_status(self, new_status: ThreadStatus) -> None:
        self._thread_status = new_status
