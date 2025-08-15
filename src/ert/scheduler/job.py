from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import time
import warnings
from collections import Counter
from contextlib import suppress
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from lxml import etree
from opentelemetry.trace import Status, StatusCode

from _ert.events import (
    RealizationEvent,
    RealizationFailed,
    RealizationPending,
    RealizationResubmit,
    RealizationRunning,
    RealizationSuccess,
    RealizationTimeout,
    RealizationWaiting,
)
from ert.config import ForwardModelStep
from ert.constant_filenames import ERROR_file
from ert.storage.load_status import LoadStatus
from ert.storage.local_ensemble import forward_model_ok
from ert.storage.realization_storage_state import RealizationStorageState
from ert.trace import trace, tracer
from ert.warnings import PostSimulationWarning

from .driver import Driver, FailedSubmit

if TYPE_CHECKING:
    from ert.ensemble_evaluator import Realization

    from .scheduler import Scheduler

logger = logging.getLogger(__name__)


class JobState(StrEnum):
    WAITING = "WAITING"
    RESUBMITTING = "RESUBMITTING"
    SUBMITTING = "SUBMITTING"
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    ABORTING = "ABORTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ABORTED = "ABORTED"


FILE_VERIFICATION_LOG_TIME_THRESHOLD = 5
DISK_SYNCHRONIZATION_POLLING_INTERVAL = 1


class Job:
    """Handle to a single job scheduler job.

    Instances of this class represent a single job as submitted to a job scheduler
    (LSF, PBS, SLURM, etc.)
    """

    DEFAULT_FILE_VERIFICATION_TIMEOUT = 120
    WAIT_PERIOD_FOR_TERM_MESSAGE_TO_CANCEL = 5

    def __init__(self, scheduler: Scheduler, real: Realization) -> None:
        self.real = real
        self.state = JobState.WAITING
        self.started = asyncio.Event()
        self.exec_hosts: str = "-"
        self.returncode: asyncio.Future[int] = asyncio.Future()
        self._scheduler: Scheduler = scheduler
        self._message: str = ""
        self._requested_max_submit: int | None = None
        self._start_time: float | None = None
        self._end_time: float | None = None
        self._remaining_file_verification_time = self.DEFAULT_FILE_VERIFICATION_TIMEOUT
        self._previous_file_verification_time = self._remaining_file_verification_time
        self._started_killing_by_evaluator: bool = False
        self._was_killed_by_evaluator = asyncio.Event()

    @property
    def remaining_file_verification_time(self) -> int:
        return self._remaining_file_verification_time

    @remaining_file_verification_time.setter
    def remaining_file_verification_time(self, value: int) -> None:
        self._previous_file_verification_time = self._remaining_file_verification_time
        self._remaining_file_verification_time = value

    def elapsed_time(self) -> int:
        return (
            self._previous_file_verification_time
            - self._remaining_file_verification_time
        )

    def unschedule(self, msg: str) -> None:
        self.state = JobState.ABORTED
        self.real.run_arg.ensemble_storage.set_failure(
            self.real.iens,
            RealizationStorageState.LOAD_FAILURE,
            f"Job not scheduled due to {msg}",
        )

    @property
    def iens(self) -> int:
        return self.real.iens

    @property
    def driver(self) -> Driver:
        return self._scheduler.driver

    @property
    def running_duration(self) -> float:
        if self._start_time:
            if self._end_time:
                return self._end_time - self._start_time
            return time.time() - self._start_time
        return 0

    async def _submit_and_run_once(self, sem: asyncio.BoundedSemaphore) -> None:
        await self._send(JobState.WAITING)
        await sem.acquire()
        timeout_task: asyncio.Task[None] | None = None

        try:
            if self._scheduler.submit_sleep_state:
                await self._scheduler.submit_sleep_state.sleep_until_we_can_submit()
            await self._send(JobState.SUBMITTING)
            self.submit_time = time.time()
            try:
                await self.driver.submit(
                    self.real.iens,
                    self.real.job_script,
                    self.real.run_arg.runpath,
                    num_cpu=self.real.num_cpu,
                    realization_memory=self.real.realization_memory,
                    name=self.real.run_arg.job_name,
                    runpath=Path(self.real.run_arg.runpath),
                )
            except FailedSubmit as err:
                await self._send(JobState.FAILED)
                logger.error(f"Failed to submit: {err}")
                self.returncode.cancel()
                return

            await self._send(JobState.PENDING)
            await self.started.wait()
            self._start_time = time.time()
            pending_time = self._start_time - self.submit_time
            logger.info(
                f"Pending time for realization {self.iens} "
                f"was {pending_time:.2f} seconds "
                f"(num_cpu={self.real.num_cpu} "
                f"realization_memory={self.real.realization_memory})"
            )

            await self._send(JobState.RUNNING)
            if self.real.max_runtime is not None and self.real.max_runtime > 0:
                timeout_task = asyncio.create_task(self._max_runtime_task())

            await self.returncode

        except asyncio.CancelledError:
            await self._send(JobState.ABORTING)
            killed_by_evaluator = False
            if self._started_killing_by_evaluator:
                with suppress(asyncio.TimeoutError):
                    killed_by_evaluator = await asyncio.wait_for(
                        self._was_killed_by_evaluator.wait(),
                        timeout=self.WAIT_PERIOD_FOR_TERM_MESSAGE_TO_CANCEL,
                    )
            if not killed_by_evaluator:
                logger.warning(
                    f"Realization {self.iens} was not killed gracefully by "
                    "TERM message. Killing it with the scheduler"
                )
                await self.driver.kill(self.iens)
            else:
                logger.info(f"Realization {self.iens} was killed by the evaluator")
            with suppress(asyncio.CancelledError):
                self.returncode.cancel()
            await self._send(JobState.ABORTED)
        finally:
            if timeout_task and not timeout_task.done():
                timeout_task.cancel()
            sem.release()

    async def log_time_spent_above_threshold_waiting_for_files(
        self, method_name: str
    ) -> None:
        if self._previous_file_verification_time <= 0:
            return
        elapsed_time = self.elapsed_time()
        if self.remaining_file_verification_time <= 0:
            logger.warning(
                f"{method_name} timed out after waiting "
                f"{elapsed_time} seconds for files"
            )
        elif elapsed_time >= FILE_VERIFICATION_LOG_TIME_THRESHOLD:
            logger.warning(
                f"{method_name} spent {elapsed_time} seconds waiting for files"
            )

    @tracer.start_as_current_span(f"{__name__}.run")
    async def run(
        self,
        sem: asyncio.BoundedSemaphore,
        forward_model_ok_lock: asyncio.Lock,
        checksum_lock: asyncio.Lock,
        max_submit: int = 1,
    ) -> None:
        current_span = trace.get_current_span()
        current_span.set_attribute("ert.realization_number", self.iens)
        self._requested_max_submit = max_submit
        for attempt in range(max_submit):
            await self._submit_and_run_once(sem)

            if self.returncode.cancelled() or self._scheduler._cancelled:
                break

            if self.returncode.result() == 0:
                if self._scheduler._manifest_queue is not None:
                    await self._verify_checksum(checksum_lock)
                    await self.log_time_spent_above_threshold_waiting_for_files(
                        method_name=self._verify_checksum.__name__
                    )

                async with forward_model_ok_lock:
                    await self._handle_finished_forward_model()

                if not self._scheduler.warnings_extracted:
                    self._scheduler.warnings_extracted = True
                    self.remaining_file_verification_time = (
                        await log_warnings_from_forward_model(
                            self.real,
                            job_submission_time=self.submit_time,
                            timeout_seconds=self.remaining_file_verification_time,
                        )
                    )
                    await self.log_time_spent_above_threshold_waiting_for_files(
                        method_name=log_warnings_from_forward_model.__name__
                    )
                break

            if attempt < max_submit - 1:
                message = (
                    f"Realization {self.iens} failed, "
                    f"resubmitting for attempt {attempt + 2} of {max_submit}"
                )
                logger.warning(message)
                self.returncode = asyncio.Future()
                self.started.clear()
                await self._send(JobState.RESUBMITTING)
            else:
                current_span.set_status(Status(StatusCode.ERROR))
                await self._send(JobState.FAILED)

    async def _max_runtime_task(self) -> None:
        assert self.real.max_runtime is not None
        await asyncio.sleep(self.real.max_runtime)
        timeout_event = RealizationTimeout(
            real=str(self.iens), ensemble=self._scheduler._ens_id
        )
        assert self._scheduler._events is not None
        await self._scheduler._events.put(timeout_event)
        logger.warning(
            f"Realization {self.iens} stopped due to "
            f"MAX_RUNTIME={self.real.max_runtime} seconds"
        )
        self.returncode.cancel()

    async def _verify_checksum(
        self,
        checksum_lock: asyncio.Lock,
        timeout: int | None = None,  # noqa: ASYNC109
    ) -> None:
        if timeout is None:
            timeout = self.DEFAULT_FILE_VERIFICATION_TIMEOUT
        # Wait for job runpath to be in the checksum dictionary
        runpath = self.real.run_arg.runpath
        while runpath not in self._scheduler.checksum:
            if timeout <= 0:
                break
            timeout -= 1
            await asyncio.sleep(1)

        checksum = self._scheduler.checksum.get(runpath)
        if checksum is None:
            logger.warning(f"Checksum information not received for {runpath}")
            self.remaining_file_verification_time = timeout
            return

        errors = "\n".join(
            [info["error"] for info in checksum.values() if "error" in info]
        )
        if errors:
            logger.error(errors)

        valid_checksums = [info for info in checksum.values() if "error" not in info]

        # Wait for files in checksum
        while not all(Path(info["path"]).exists() for info in valid_checksums):
            if timeout <= 0:
                break
            timeout -= DISK_SYNCHRONIZATION_POLLING_INTERVAL
            logger.debug("Waiting for disk synchronization")
            await asyncio.sleep(DISK_SYNCHRONIZATION_POLLING_INTERVAL)
        async with checksum_lock:
            for info in valid_checksums:
                file_path = Path(info["path"])
                expected_md5sum = info.get("md5sum")
                if file_path.exists() and expected_md5sum:
                    actual_md5sum = hashlib.md5(file_path.read_bytes()).hexdigest()
                    if expected_md5sum == actual_md5sum:
                        logger.debug(f"File {file_path} checksum successful.")
                    else:
                        logger.warning(
                            f"File {file_path} checksum verification failed."
                        )
                elif file_path.exists() and expected_md5sum is None:
                    logger.warning(f"Checksum not received for file {file_path}")
                else:
                    logger.error(f"Disk synchronization failed for {file_path}")
        self.remaining_file_verification_time = timeout

    async def _handle_finished_forward_model(self) -> None:
        callback_status, status_msg = await forward_model_ok(
            run_path=self.real.run_arg.runpath,
            realization=self.real.run_arg.iens,
            iter_=self.real.run_arg.itr,
            ensemble=self.real.run_arg.ensemble_storage,
        )
        if self._message:
            self._message = status_msg
        else:
            self._message += f"\nstatus from done callback: {status_msg}"

        if callback_status == LoadStatus.LOAD_SUCCESSFUL:
            await self._send(JobState.COMPLETED)
        else:
            assert callback_status == LoadStatus.LOAD_FAILURE
            await self._send(JobState.FAILED)

    async def _handle_failure(self) -> None:
        assert self._requested_max_submit is not None

        error_msg = (
            f"Realization: {self.real.run_arg.iens} "
            f"failed after reaching max submit ({self._requested_max_submit}):"
            f"\n\t{self._message}"
        )

        if msg := self.driver._job_error_message_by_iens.get(self.iens, ""):
            error_msg += f"\nDriver reported: {msg}"

        error_msg += self.driver.read_stdout_and_stderr_files(
            self.real.run_arg.runpath, self.real.run_arg.job_name
        )

        with contextlib.suppress(OSError):
            self.real.run_arg.ensemble_storage.set_failure(
                self.real.run_arg.iens, RealizationStorageState.LOAD_FAILURE, error_msg
            )
        logger.error(error_msg)
        log_info_from_exit_file(Path(self.real.run_arg.runpath) / ERROR_file)

    async def _handle_aborted(self) -> None:
        self.real.run_arg.ensemble_storage.set_failure(
            self.real.run_arg.iens,
            RealizationStorageState.LOAD_FAILURE,
            "Job cancelled",
        )
        log_info_from_exit_file(Path(self.real.run_arg.runpath) / ERROR_file)

    async def _send(self, state: JobState) -> None:
        event: RealizationEvent | None = None
        match state:
            case JobState.WAITING | JobState.SUBMITTING:
                event = RealizationWaiting(real=str(self.iens))
            case JobState.RESUBMITTING:
                event = RealizationResubmit(real=str(self.iens))
            case JobState.PENDING:
                event = RealizationPending(real=str(self.iens))
            case JobState.RUNNING:
                event = RealizationRunning(real=str(self.iens))
            case JobState.FAILED:
                event = RealizationFailed(real=str(self.iens))
                event.message = self._message
                await self._handle_failure()
            case JobState.ABORTING:
                event = RealizationFailed(real=str(self.iens))
            case JobState.ABORTED:
                event = RealizationFailed(real=str(self.iens))
                await self._handle_aborted()
            case JobState.COMPLETED:
                event = RealizationSuccess(real=str(self.iens))
                self._end_time = time.time()
                await self._scheduler.completed_jobs.put(self.iens)

        self.state = state
        if event is not None:
            event.ensemble = self._scheduler._ens_id
            event.queue_event_type = state
            event.exec_hosts = self.exec_hosts

            await self._scheduler._events.put(event)


def log_info_from_exit_file(exit_file_path: Path) -> None:
    if not exit_file_path.exists():
        return
    try:
        exit_file = etree.parse(exit_file_path)
    except OSError as err:
        logger.error(
            f"Realization failed and the XML error file {exit_file_path} "
            f"could not not be read: {err}"
        )
        return
    except etree.XMLSyntaxError:
        raw_xml_contents = exit_file_path.read_text(encoding="utf-8", errors="ignore")
        logger.error(
            "Realization failed with an invalid "
            f"XML ERROR file, contents '{raw_xml_contents}'"
        )
        return
    logger.error(
        f"Step {exit_file.findtext('step')} failed with: "
        f"'{exit_file.findtext('reason')}'\n\t"
        f"stderr file: '{exit_file.findtext('stderr_file')}',\n\t"
        f"its contents:{exit_file.findtext('stderr')}"
    )


async def log_warnings_from_forward_model(
    real: Realization,
    job_submission_time: float,
    timeout_seconds: int = Job.DEFAULT_FILE_VERIFICATION_TIMEOUT,
) -> int:
    """Parse all stdout and stderr files from running the forward model
    for anything that looks like a Warning, and log it.

    This is not a critical task to perform, but it is critical not to crash
    during this process.

    Args:
        real: The realization to look for warnings in
        job_submission_time: The time the job for the given realization was last
            started. Files not changed after the job started are not read. There is
            a retry used to wait for file sync.
        timeout_seconds: Time to wait for stdout and stderr to appear if missing.
    Returns:
        The seconds left of the given timeout_seconds.
    """

    max_length = 2048  # Lines will be truncated in length when logged

    def line_contains_warning(line: str) -> bool:
        return (
            "Warning:" in line
            or "FutureWarning" in line
            or "DeprecationWarning" in line
            or "UserWarning" in line
            or ":WARNING:" in line
            or "- WARNING - " in line
            or "- ERROR - " in line
        )

    async def log_warnings_from_file(  # noqa
        file: Path, iens: int, step: ForwardModelStep, step_idx: int, filetype: str
    ) -> None:
        captured: list[str] = []
        for line in file.read_text(encoding="utf-8").splitlines():
            if line_contains_warning(line):
                captured.append(line[:max_length])

        for line, counter in Counter(captured).items():
            warning_msg = (
                f"Realization {iens} step {step.name}.{step_idx} "
                f"warned {counter} time(s) in {filetype}: {line}"
            )
            warnings.warn(warning_msg, PostSimulationWarning, stacklevel=2)
            logger.warning(warning_msg)

    async def wait_for_file(file_path: Path, _timeout: int) -> int:
        if _timeout <= 0:
            return 0
        remaining_timeout = _timeout
        for _ in range(_timeout):
            if not (
                file_path.exists() and file_path.stat().st_mtime >= job_submission_time
            ):
                remaining_timeout -= 1
                await asyncio.sleep(1)
            else:
                break
        return remaining_timeout

    with suppress(KeyError):
        runpath = Path(real.run_arg.runpath)
        for step_idx, step in enumerate(real.fm_steps):
            for std_file_name, file_type in [
                (step.stdout_file, "stdout"),
                (step.stderr_file, "stderr"),
            ]:
                if std_file_name is not None:
                    std_path = runpath / f"{std_file_name}.{step_idx}"
                    timeout_seconds = await wait_for_file(std_path, timeout_seconds)

                    if timeout_seconds <= 0:
                        break

                    await log_warnings_from_file(
                        std_path, real.iens, step, step_idx, file_type
                    )
            if timeout_seconds <= 0:
                break
    return timeout_seconds
