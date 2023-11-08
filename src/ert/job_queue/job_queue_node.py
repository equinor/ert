from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from threading import Lock, Semaphore, Thread
from typing import TYPE_CHECKING, Callable, Optional

from cwrap import BaseCClass

from ert._clib.queue import (  # pylint: disable=import-error
    _get_submit_attempt,
    _kill,
    _refresh_status,
    _submit,
)
from ert.callbacks import forward_model_ok
from ert.load_status import LoadStatus

from ..realization_state import RealizationState
from . import ResPrototype
from .job_status import JobStatus
from .submit_status import SubmitStatus
from .thread_status import ThreadStatus

if TYPE_CHECKING:
    from ..run_arg import RunArg
    from .queue import Driver

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


class JobQueueNode:
    def __init__(
        self,
        job_script: str,
        num_cpu: int,
        status_file: str,
        exit_file: str,
        run_arg: "RunArg",
        max_runtime: Optional[int] = None,
        callback_timeout: Optional[Callable[[int], None]] = None,
    ):
        self.callback_timeout = callback_timeout
        self.run_arg = run_arg

        self.thread_status: ThreadStatus = ThreadStatus.READY
        self._thread: Optional[Thread] = None
        self._mutex = Lock()
        self._tried_killing = 0

        self._max_runtime = max_runtime
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._timed_out = False
        self._status: JobStatus = JobStatus.UNKNOWN
        self._status_msg = ""

        # c-struct attributes
        self._status_file: str = status_file
        self._exit_file: str = exit_file
        self._run_cmd: str = job_script
        self._job_name: str = run_arg.job_name
        self._run_path: str = run_arg.runpath
        self._num_cpu: int = num_cpu
        self._queue_index: int = 0
        self._submit_attempt: int = 0
        self._confirmed_running: bool = False
        self._fail_message: Optional[str] = None
        self._error_message: Optional[str] = None

    def __str__(self) -> str:
        return (
            f"JobNode: Name:{self.run_arg.job_name}, Status: {self.queue_status}, "
            f"Timed_out: {self.timed_out}, "
            f"Submit_attempt: {self.submit_attempt}"
        )

    @property
    def run_path(self) -> str:
        return self._run_path

    @property
    def timed_out(self) -> bool:
        with self._mutex:
            return self._timed_out

    @property
    def submit_attempt(self) -> int:
        # return _get_submit_attempt(self)
        return self._submit_attempt

    def refresh_status(self, driver: "Driver") -> JobStatus:
        if self.queue_status == JobStatus.RUNNING and not self._confirmed_running:
            self._confirmed_running = Path(self._status_file).exists()
            if not self._confirmed_running:
                MAX_CONFIRMED_WAIT = 10 * 60
                if self.runtime > MAX_CONFIRMED_WAIT:
                    logger.error(
                        f"max_confirm_wait {MAX_CONFIRMED_WAIT} has passed since sim_start"
                        f"without success; {self._job_name} is assumed dead (attempt {self._submit_attempt})"
                    )
                    self.queue_status = JobStatus.DO_KILL_NODE_FAILURE
        if self.is_running():
            try:
                self._status = driver.get_status(self)
            except Exception:
                self.queue_status = JobStatus.STATUS_FAILURE
            if self.queue_status == JobStatus.EXIT:
                with open(self._exit_file, "r") as exit_file:
                    # needs to be parsed properly
                    self._fail_message = exit_file.readlines()[0]
        if self._fail_message and not self._error_message:
            self._error_message = self._fail_message
        return self.queue_status

    @property
    def queue_status(self) -> JobStatus:
        # return self._get_status()
        return self._status

    @queue_status.setter
    def queue_status(self, value: JobStatus) -> None:
        self._status = value

    def submit(self, driver: "Driver") -> SubmitStatus:
        return driver.submit(self)

    def run_done_callback(self) -> Optional[LoadStatus]:
        callback_status, status_msg = forward_model_ok(self.run_arg)
        if callback_status == LoadStatus.LOAD_SUCCESSFUL:
            self.queue_status = JobStatus.SUCCESS  # type: ignore
        elif callback_status == LoadStatus.TIME_MAP_FAILURE:
            self.queue_status = JobStatus.FAILED  # type: ignore
        else:
            self.queue_status = JobStatus.EXIT  # type: ignore
        if self._status_msg != "":
            self._status_msg = status_msg
        else:
            self._status_msg += f"\nstatus from done callback: {status_msg}"
        return callback_status

    def run_timeout_callback(self) -> None:
        if self.callback_timeout:
            self.callback_timeout(self.run_arg.iens)

    def run_exit_callback(self) -> None:
        self.run_arg.ensemble_storage.state_map[
            self.run_arg.iens
        ] = RealizationState.LOAD_FAILURE

    def is_running(self, given_status: Optional[JobStatus] = None) -> bool:
        status = given_status or self.queue_status
        return status in (
            JobStatus.PENDING,
            JobStatus.SUBMITTED,
            JobStatus.RUNNING,
            JobStatus.UNKNOWN,
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
        if submit_status is not SubmitStatus.OK:
            self.queue_status = JobStatus.DONE  #  type: ignore

        end_status = self._poll_until_done(driver)
        self._handle_end_status(driver, pool_sema, end_status, max_submit)

    def _poll_until_done(self, driver: "Driver") -> JobStatus:
        current_status = self._poll_queue_status(driver)
        backoff = _BackoffFunction()
        # in the following loop, we increase the sleep time between loop iterations as
        # long running realizations do not change state often, and too frequent querying
        # with many realizations starves other threads for resources.
        while self.is_running(current_status):
            if self._start_time is None and current_status == JobStatus.RUNNING:
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

    RESUBMIT_STATES = [JobStatus.EXIT]
    DONE_STATES = [
        JobStatus.SUCCESS,
        JobStatus.IS_KILLED,
        JobStatus.DO_KILL_NODE_FAILURE,
    ]
    FAILURE_STATES = [JobStatus.FAILED]

    def _handle_end_status(
        self,
        driver: "Driver",
        pool_sema: Semaphore,
        end_status: JobStatus,
        max_submit: int,
    ) -> None:
        with self._mutex:
            if end_status == JobStatus.DONE:
                with pool_sema:
                    logger.info(
                        f"Realization: {self.run_arg.iens} complete, "
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
                        f"Realization: {self.run_arg.iens} "
                        f"failed with: {self._status_msg}, resubmitting"
                    )
                    self._transition_status(ThreadStatus.READY, current_status)
                else:
                    self._transition_to_failure(
                        message=f"Realization: {self.run_arg.iens} "
                        "failed after reaching max submit"
                        f" ({max_submit}):\n\t{self._status_msg}"
                    )
            elif current_status in self.FAILURE_STATES:
                self._transition_to_failure(
                    message=f"Realization: {self.run_arg.iens} "
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
            queue_status=JobStatus.FAILED,  # type: ignore
        )

    def _transition_status(
        self,
        thread_status: ThreadStatus,
        queue_status: JobStatus,
    ) -> None:
        self.queue_status = queue_status
        if thread_status == ThreadStatus.DONE and queue_status != JobStatus.SUCCESS:
            self.run_exit_callback()
        self.thread_status = thread_status

    def _kill(self, driver: "Driver") -> None:
        _kill(self, driver)
        self._tried_killing += 1

    def run(self, driver: "Driver", pool_sema: Semaphore, max_submit: int = 2) -> None:
        # Prevent multiple threads working on the same object
        self.wait_for()
        # Do not start if already kill signal is sent
        if self.thread_status == ThreadStatus.STOPPING:
            self.thread_status = ThreadStatus.DONE
            return

        self.thread_status = ThreadStatus.RUNNING
        self._start_time = None
        self._thread = Thread(
            target=self._job_monitor, args=(driver, pool_sema, max_submit)
        )
        self._thread.start()

    def stop(self) -> None:
        with self._mutex:
            if self.thread_status == ThreadStatus.RUNNING:
                self.thread_status = ThreadStatus.STOPPING
            elif self.thread_status == ThreadStatus.READY:
                # clean-up to get the correct status after being stopped by user
                self.thread_status = ThreadStatus.DONE
                self.queue_status = JobStatus.FAILED  # type: ignore

            assert self.thread_status in [
                ThreadStatus.DONE,
                ThreadStatus.STOPPING,
                ThreadStatus.FAILED,
            ]

    def wait_for(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            self._thread.join()
