from __future__ import annotations

import logging
import multiprocessing as mp
import random
import sys
import time
import traceback
from ctypes import c_int
from threading import Lock, Semaphore, Thread
from typing import TYPE_CHECKING, Any, Optional, Tuple

from cwrap import BaseCClass
from ecl.util.util import StringList

from ert._clib.queue import (  # pylint: disable=import-error
    _get_submit_attempt,
    _kill,
    _refresh_status,
    _submit,
)
from ert.load_status import LoadStatus
from ert.realization_state import RealizationState

from . import ResPrototype
from .job_status_type_enum import JobStatusType
from .job_submit_status_type_enum import JobSubmitStatusType
from .thread_status_type_enum import ThreadStatus

if TYPE_CHECKING:
    from multiprocessing.sharedctypes import Synchronized, SynchronizedString

    from ert.callbacks import Callback, CallbackArgs

    from .driver import Driver

logger = logging.getLogger(__name__)


class _BackoffFunction:
    def __init__(
        self,
        time_sleep_seconds: int = 1,
        max_sleep_seconds: int = 30,
        time_until_longer_sleep_seconds: int = 30,
        use_random_sleep_offset: bool = False,
    ) -> None:
        self.max_sleep_seconds = max_sleep_seconds
        self.time_until_longer_sleep_seconds = time_until_longer_sleep_seconds
        self.use_random_sleep_offset = use_random_sleep_offset
        self.time_sleep_seconds = time_sleep_seconds

    def __call__(
        self,
        elapsed_time: float,
    ) -> int:
        if (
            self.time_sleep_seconds != self.max_sleep_seconds
            and elapsed_time > self.time_until_longer_sleep_seconds
        ):
            self.time_sleep_seconds = self.max_sleep_seconds
            self.use_random_sleep_offset = True
        return self.time_sleep_seconds + self.use_random_sleep_offset * random.randint(
            -5, 5
        )


# pylint: disable=too-many-instance-attributes
class JobQueueNode(BaseCClass):  # type: ignore
    TYPE_NAME = "job_queue_node"

    _alloc = ResPrototype(
        "void* job_queue_node_alloc(char*,"
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
    _get_status = ResPrototype(
        "job_status_type_enum job_queue_node_get_status(job_queue_node)"
    )
    _set_queue_status = ResPrototype(
        "void job_queue_node_set_status(job_queue_node, job_status_type_enum)"
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
        return _get_submit_attempt(self)  # type: ignore

    def _poll_queue_status(self, driver: "Driver") -> JobStatusType:
        result, msg = _refresh_status(self, driver)
        if msg is not None:
            self._status_msg = msg
        return JobStatusType(result)

    @property
    def status(self) -> JobStatusType:
        return self._get_status()  # type: ignore

    @property
    def thread_status(self) -> ThreadStatus:
        return self._thread_status

    def submit(self, driver: "Driver") -> JobSubmitStatusType:
        return JobSubmitStatusType(_submit(self, driver))

    def run_done_callback(self) -> Optional[LoadStatus]:
        if sys.platform == "linux":
            callback_status, status_msg = self.run_done_callback_forking()
        else:
            try:
                callback_status, status_msg = self.done_callback_function(
                    *self.callback_arguments
                )
            except Exception as err:  # pylint: disable=broad-exception-caught
                exception_with_stack = "".join(
                    traceback.format_exception(type(err), err, err.__traceback__)
                )
                error_message = (
                    "got exception while running forward_model_ok "
                    f"callback:\n{exception_with_stack}"
                )
                print(error_message)
                logger.exception(err)
                callback_status = LoadStatus.LOAD_FAILURE
                status_msg = error_message
        if callback_status == LoadStatus.LOAD_SUCCESSFUL:
            self._set_queue_status(JobStatusType.JOB_QUEUE_SUCCESS)
        elif callback_status == LoadStatus.TIME_MAP_FAILURE:
            self._set_queue_status(JobStatusType.JOB_QUEUE_FAILED)
        else:
            self._set_queue_status(JobStatusType.JOB_QUEUE_EXIT)
        if self._status_msg != "":
            self._status_msg = status_msg
        else:
            self._status_msg += f"\nstatus from done callback: {status_msg}"
        return callback_status

    def run_timeout_callback(self) -> None:
        if self.callback_timeout:
            self.callback_timeout(*self.callback_arguments)

    # this function only works on systems where multiprocessing.Process uses forking
    def run_done_callback_forking(self) -> Tuple[LoadStatus, str]:
        # status_msg has a maximum length of 1024 bytes.
        # the size is immutable after creation due to being backed by a c array.
        status_msg: "SynchronizedString" = mp.Array("c", b" " * 1024)  # type: ignore
        callback_status: "Synchronized[c_int]" = mp.Value("i", 2)  # type: ignore
        pcontext = ProcessWithException(
            target=self.done_callback_wrapper,
            kwargs={
                "callback_arguments": self.callback_arguments,
                "callback_status_shared": callback_status,
                "status_msg_shared": status_msg,
            },
        )
        pcontext.start()
        try:
            pcontext.wait_and_throw_if_exception()
        except Exception as err:  # pylint: disable=broad-exception-caught
            exception_with_stack = "".join(
                traceback.format_exception(type(err), err, err.__traceback__)
            )
            error_message = (
                "got exception while running forward_model_ok "
                f"callback:\n{exception_with_stack}"
            )
            print(error_message)
            logger.exception(err)
        pcontext.join()

        load_status = LoadStatus(callback_status.value)

        # this step was added because the state_map update in
        #      forward_model_ok does not propagate from the spawned process.
        run_arg = self.callback_arguments[0]
        run_arg.ensemble_storage.state_map[run_arg.iens] = (
            RealizationState.HAS_DATA
            if load_status == LoadStatus.LOAD_SUCCESSFUL
            else RealizationState.LOAD_FAILURE
        )

        return load_status, status_msg.value.decode("utf-8")

    def done_callback_wrapper(
        self,
        callback_arguments: CallbackArgs,
        callback_status_shared: "Synchronized[c_int]",
        status_msg_shared: "SynchronizedString",
    ) -> None:
        callback_status: Optional[LoadStatus]
        status_msg: str
        callback_status, status_msg = self.done_callback_function(*callback_arguments)

        if callback_status is not None:
            callback_status_shared.value = callback_status.value  # type: ignore
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

    def _job_monitor(
        self, driver: "Driver", pool_sema: Semaphore, max_submit: int
    ) -> None:
        submit_status = self.submit(driver)
        if submit_status is not JobSubmitStatusType.SUBMIT_OK:
            self._set_queue_status(JobStatusType.JOB_QUEUE_DONE)

        end_status = self._poll_until_done(driver)
        self._handle_end_status(driver, pool_sema, end_status, max_submit)

    def _poll_until_done(self, driver: Driver) -> JobStatusType:
        current_status = self._poll_queue_status(driver)
        backoff = _BackoffFunction()
        # in the following loop, we increase the sleep time between loop iterations as
        # long running realizations do not change state often, and too frequent querying
        # with many realizations starves other threads for resources.
        while self.is_running(current_status):
            if (
                self._start_time is None
                and current_status == JobStatusType.JOB_QUEUE_RUNNING
            ):
                self._start_time = time.time()
            if self._start_time is not None:
                elapsed_time = time.time() - self._start_time
            else:
                elapsed_time = 0.0
            time.sleep(backoff(elapsed_time))
            if self._exceeded_allowed_runtime():
                self._kill(driver)
                self._log_kill_timeout_status()
                self.run_timeout_callback()
                with self._mutex:
                    self._timed_out = True
            elif self.thread_status == ThreadStatus.STOPPING:
                self._kill(driver)
                self._log_kill_thread_stopping_status()

            current_status = self._poll_queue_status(driver)
        self._end_time = time.time()
        return current_status

    def _exceeded_allowed_runtime(self) -> bool:
        return self._max_runtime is not None and self.runtime >= self._max_runtime

    def _log_kill_timeout_status(self) -> None:
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

    def _log_kill_thread_stopping_status(self) -> None:
        if self._tried_killing == 1:
            logger.error(f"Killing job in {self.run_path} ({self.thread_status}).")

    RESUBMIT_STATES = [JobStatusType.JOB_QUEUE_EXIT]
    DONE_STATES = [
        JobStatusType.JOB_QUEUE_SUCCESS,
        JobStatusType.JOB_QUEUE_IS_KILLED,
        JobStatusType.JOB_QUEUE_DO_KILL_NODE_FAILURE,
    ]
    FAILURE_STATES = [JobStatusType.JOB_QUEUE_FAILED]

    def _handle_end_status(
        self,
        driver: Driver,
        pool_sema: Semaphore,
        end_status: JobStatusType,
        max_submit: int,
    ) -> None:
        with self._mutex:
            if end_status == JobStatusType.JOB_QUEUE_DONE:
                with pool_sema:
                    logger.info(
                        f"Realization: {self.callback_arguments[0].iens} complete, "
                        "starting to load results"
                    )
                    self.run_done_callback()

            # refresh cached status after running the callback
            current_status = self._poll_queue_status(driver)

            if current_status in self.DONE_STATES:
                self._transition_status(ThreadStatus.DONE, current_status)
            elif current_status in self.RESUBMIT_STATES:
                if self.submit_attempt < max_submit:
                    logger.warning(
                        f"Realization: {self.callback_arguments[0].iens} "
                        f"failed with: {self._status_msg}, resubmitting"
                    )
                    self._transition_status(ThreadStatus.READY, current_status)
                else:
                    self._transition_to_failure(
                        message=f"Realization: {self.callback_arguments[0].iens} "
                        "failed after reaching max submit"
                        f" ({max_submit}):\n\t{self._status_msg}"
                    )
            elif current_status in self.FAILURE_STATES:
                self._transition_to_failure(
                    message=f"Realization: {self.callback_arguments[0].iens} "
                    f"failed with: {self._status_msg}"
                )
            else:
                self._transition_status(ThreadStatus.FAILED, current_status)
                raise ValueError(
                    f"Unexpected job status after running: {current_status}"
                )

    def _transition_to_failure(self, message: str) -> None:
        logger.error(message)
        self._transition_status(
            thread_status=ThreadStatus.DONE,
            queue_status=JobStatusType.JOB_QUEUE_FAILED,  # type: ignore
        )

    def _transition_status(
        self,
        thread_status: ThreadStatus,
        queue_status: JobStatusType,
    ) -> None:
        self._set_queue_status(queue_status)
        if (
            thread_status == ThreadStatus.DONE
            and queue_status != JobStatusType.JOB_QUEUE_SUCCESS
        ):
            self.run_exit_callback()
        self._set_thread_status(thread_status)

    def _kill(self, driver: "Driver") -> None:
        _kill(self, driver)
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
                self._set_queue_status(JobStatusType.JOB_QUEUE_FAILED)

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


class ProcessWithException(mp.Process):
    """Used to run something in a subprocess, and capture exceptions. In order to catch
    an exception, wait_and_throw_if_exception should be tried - before one joins the
    process!"""

    def __init__(self, *args: Any, **kwargs: Any):
        mp.Process.__init__(self, *args, **kwargs)
        self._parent_connection, self._child_connection = mp.Pipe(False)
        self._exception = None

    def run(self) -> None:
        try:
            mp.Process.run(self)
            self._child_connection.send(None)
        except Exception as err:  # pylint: disable=broad-exception-caught
            self._child_connection.send(err)

    def wait_and_throw_if_exception(self) -> None:
        exception = self._parent_connection.recv()
        if exception:
            raise exception
