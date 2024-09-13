from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from contextlib import suppress
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from lxml import etree
from pydantic_core._pydantic_core import ValidationError

from _ert.events import Id, RealizationTimeout, event_from_dict
from ert.callbacks import forward_model_ok
from ert.constant_filenames import ERROR_file
from ert.load_status import LoadStatus
from ert.storage.realization_storage_state import RealizationStorageState

from .driver import Driver

if TYPE_CHECKING:
    from ert.ensemble_evaluator import Realization

    from .scheduler import Scheduler

logger = logging.getLogger(__name__)


class JobState(str, Enum):
    WAITING = "WAITING"
    SUBMITTING = "SUBMITTING"
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    ABORTING = "ABORTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ABORTED = "ABORTED"


_queue_jobstate_event_type = {
    JobState.WAITING: Id.REALIZATION_WAITING,
    JobState.SUBMITTING: Id.REALIZATION_WAITING,
    JobState.PENDING: Id.REALIZATION_PENDING,
    JobState.RUNNING: Id.REALIZATION_RUNNING,
    JobState.ABORTING: Id.REALIZATION_FAILURE,
    JobState.COMPLETED: Id.REALIZATION_SUCCESS,
    JobState.FAILED: Id.REALIZATION_FAILURE,
    JobState.ABORTED: Id.REALIZATION_FAILURE,
}


class Job:
    """Handle to a single job scheduler job.

    Instances of this class represent a single job as submitted to a job scheduler
    (LSF, PBS, SLURM, etc.)
    """

    def __init__(self, scheduler: Scheduler, real: Realization) -> None:
        self.real = real
        self.state = JobState.WAITING
        self.started = asyncio.Event()
        self.returncode: asyncio.Future[int] = asyncio.Future()
        self._aborted = False
        self._scheduler: Scheduler = scheduler
        self._callback_status_msg: str = ""
        self._requested_max_submit: Optional[int] = None
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None

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
        timeout_task: Optional[asyncio.Task[None]] = None

        try:
            if self._scheduler.submit_sleep_state:
                await self._scheduler.submit_sleep_state.sleep_until_we_can_submit()
            await self._send(JobState.SUBMITTING)
            await self.driver.submit(
                self.real.iens,
                self.real.job_script,
                self.real.run_arg.runpath,
                num_cpu=self.real.num_cpu,
                realization_memory=self.real.realization_memory,
                name=self.real.run_arg.job_name,
                runpath=Path(self.real.run_arg.runpath),
            )

            await self._send(JobState.PENDING)
            await self.started.wait()
            self._start_time = time.time()

            await self._send(JobState.RUNNING)
            if self.real.max_runtime is not None and self.real.max_runtime > 0:
                timeout_task = asyncio.create_task(self._max_runtime_task())

            await self.returncode

        except asyncio.CancelledError:
            await self._send(JobState.ABORTING)
            await self.driver.kill(self.iens)
            with suppress(asyncio.CancelledError):
                self.returncode.cancel()
            await self._send(JobState.ABORTED)
        finally:
            if timeout_task and not timeout_task.done():
                timeout_task.cancel()
            sem.release()

    async def run(
        self,
        sem: asyncio.BoundedSemaphore,
        forward_model_ok_lock: asyncio.Lock,
        max_submit: int = 1,
    ) -> None:
        self._requested_max_submit = max_submit
        for attempt in range(max_submit):
            await self._submit_and_run_once(sem)

            if self.returncode.cancelled() or self._scheduler._cancelled:
                break

            if self.returncode.result() == 0:
                async with forward_model_ok_lock:
                    if self._scheduler._manifest_queue is not None:
                        await self._verify_checksum()
                    await self._handle_finished_forward_model()
                break

            if attempt < max_submit - 1:
                message = (
                    f"Realization {self.iens} failed, "
                    f"resubmitting for attempt {attempt+2} of {max_submit}"
                )
                logger.warning(message)
                self.returncode = asyncio.Future()
                self.started.clear()
            else:
                await self._send(JobState.FAILED)

    async def _max_runtime_task(self) -> None:
        assert self.real.max_runtime is not None
        await asyncio.sleep(self.real.max_runtime)
        timeout_event = RealizationTimeout(
            real=str(self.iens), ensemble=self._scheduler._ens_id
        )
        assert self._scheduler._events is not None
        await self._scheduler._events.put(timeout_event)
        logger.error(
            f"Realization {self.iens} stopped due to MAX_RUNTIME={self.real.max_runtime} seconds"
        )
        self.returncode.cancel()

    async def _verify_checksum(self, timeout: int = 120) -> None:
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
            timeout -= 1
            logger.debug("Waiting for disk synchronization")
            await asyncio.sleep(1)

        for info in valid_checksums:
            file_path = Path(info["path"])
            expected_md5sum = info.get("md5sum")
            if file_path.exists() and expected_md5sum:
                actual_md5sum = hashlib.md5(file_path.read_bytes()).hexdigest()
                if expected_md5sum == actual_md5sum:
                    logger.debug(f"File {file_path} checksum successful.")
                else:
                    logger.warning(f"File {file_path} checksum verification failed.")
            elif file_path.exists() and expected_md5sum is None:
                logger.warning(f"Checksum not received for file {file_path}")
            else:
                logger.error(f"Disk synchronization failed for {file_path}")

    async def _handle_finished_forward_model(self) -> None:
        callback_status, status_msg = await forward_model_ok(self.real.run_arg)
        if self._callback_status_msg:
            self._callback_status_msg = status_msg
        else:
            self._callback_status_msg += f"\nstatus from done callback: {status_msg}"

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
            f"\n\t{self._callback_status_msg}"
        )

        if msg := self.driver._job_error_message_by_iens.get(self.iens, ""):
            error_msg += f"\nDriver reported: {msg}"

        error_msg += self.driver.read_stdout_and_stderr_files(
            self.real.run_arg.runpath, self.real.run_arg.job_name
        )

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
        event_dict: Dict[str, Any] = {
            "ensemble": self._scheduler._ens_id,
            "event_type": _queue_jobstate_event_type[state],
            "queue_event_type": state,
            "real": str(self.iens),
        }
        self.state = state
        if state == JobState.FAILED:
            event_dict["callback_status_message"] = self._callback_status_msg
            await self._handle_failure()

        elif state == JobState.ABORTED:
            await self._handle_aborted()

        elif state == JobState.COMPLETED:
            self._end_time = time.time()
            await self._scheduler.completed_jobs.put(self.iens)

        try:
            msg = event_from_dict(event_dict)
        except ValidationError:
            raise
        await self._scheduler._events.put(msg)


def log_info_from_exit_file(exit_file_path: Path) -> None:
    if not exit_file_path.exists():
        return
    try:
        exit_file = etree.parse(exit_file_path)
    except etree.XMLSyntaxError:
        raw_xml_contents = exit_file_path.read_text(encoding="utf-8", errors="ignore")
        logger.error(
            f"job failed with invalid XML ERROR file, contents '{raw_xml_contents}'"
        )
        return
    filecontents: List[str] = []
    for element in ["job", "reason", "stderr_file", "stderr"]:
        filecontents.append(str(exit_file.findtext(element)))
    logger.error(
        "job {} failed with: '{}'\n\tstderr file: '{}',\n\tits contents:{}".format(
            *filecontents
        )
    )
