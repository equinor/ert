from __future__ import annotations

import asyncio
import logging
import time
import uuid
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from cloudevents.conversion import to_json
from cloudevents.http import CloudEvent
from lxml import etree

from ert.callbacks import forward_model_ok
from ert.constant_filenames import ERROR_file
from ert.job_queue.queue import _queue_state_event_type
from ert.load_status import LoadStatus
from ert.scheduler.driver import Driver
from ert.storage.realization_storage_state import RealizationStorageState

if TYPE_CHECKING:
    from ert.ensemble_evaluator._builder._realization import Realization
    from ert.scheduler.scheduler import Scheduler

logger = logging.getLogger(__name__)

# Duplicated to avoid circular imports
EVTYPE_REALIZATION_TIMEOUT = "com.equinor.ert.realization.timeout"


class State(str, Enum):
    WAITING = "WAITING"
    SUBMITTING = "SUBMITTING"
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    ABORTING = "ABORTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ABORTED = "ABORTED"


STATE_TO_LEGACY = {
    State.WAITING: "WAITING",
    State.SUBMITTING: "SUBMITTED",
    State.PENDING: "PENDING",
    State.RUNNING: "RUNNING",
    State.ABORTING: "DO_KILL",
    State.COMPLETED: "SUCCESS",
    State.FAILED: "FAILED",
    State.ABORTED: "IS_KILLED",
}


class Job:
    """Handle to a single job scheduler job.

    Instances of this class represent a single job as submitted to a job scheduler
    (LSF, PBS, SLURM, etc.)
    """

    def __init__(self, scheduler: Scheduler, real: Realization) -> None:
        self.real = real
        self.state = State.WAITING
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
        await sem.acquire()
        timeout_task: Optional[asyncio.Task[None]] = None

        try:
            if self._scheduler.submit_sleep_state:
                await self._scheduler.submit_sleep_state.sleep_until_we_can_submit()
            await self._send(State.SUBMITTING)
            await self.driver.submit(
                self.real.iens,
                self.real.job_script,
                self.real.run_arg.runpath,
                name=self.real.run_arg.job_name,
                runpath=Path(self.real.run_arg.runpath),
            )

            await self._send(State.PENDING)
            await self.started.wait()
            self._start_time = time.time()

            await self._send(State.RUNNING)
            if self.real.max_runtime is not None and self.real.max_runtime > 0:
                timeout_task = asyncio.create_task(self._max_runtime_task())

            await self.returncode

        except asyncio.CancelledError:
            await self._send(State.ABORTING)
            await self.driver.kill(self.iens)
            self.returncode.cancel()
            await self._send(State.ABORTED)
        finally:
            if timeout_task and not timeout_task.done():
                timeout_task.cancel()
            sem.release()

    async def __call__(
        self, start: asyncio.Event, sem: asyncio.BoundedSemaphore, max_submit: int = 2
    ) -> None:
        self._requested_max_submit = max_submit
        await start.wait()
        for attempt in range(max_submit):
            await self._submit_and_run_once(sem)

            if self.returncode.cancelled():
                break

            if self.returncode.result() == 0:
                await self._handle_finished_forward_model()
                break

            if attempt < max_submit - 1:
                message = (
                    f"Realization {self.iens} failed, "
                    f"resubmitting for attempt {attempt+2} of {max_submit}"
                )
                logger.warning(message)
                self.returncode = asyncio.Future()
                self.started = asyncio.Event()
            else:
                await self._send(State.FAILED)

    async def _max_runtime_task(self) -> None:
        assert self.real.max_runtime is not None
        await asyncio.sleep(self.real.max_runtime)
        timeout_event = CloudEvent(
            {
                "type": EVTYPE_REALIZATION_TIMEOUT,
                "source": f"/ert/ensemble/{self._scheduler._ens_id}/real/{self.iens}",
                "id": str(uuid.uuid1()),
            }
        )
        assert self._scheduler._events is not None
        await self._scheduler._events.put(to_json(timeout_event))
        logger.error(
            f"Realization {self.iens} stopped due to MAX_RUNTIME={self.real.max_runtime} seconds"
        )
        self.returncode.cancel()

    async def _handle_finished_forward_model(self) -> None:
        callback_status, status_msg = forward_model_ok(self.real.run_arg)
        if self._callback_status_msg:
            self._callback_status_msg = status_msg
        else:
            self._callback_status_msg += f"\nstatus from done callback: {status_msg}"

        if callback_status == LoadStatus.LOAD_SUCCESSFUL:
            await self._send(State.COMPLETED)
        else:
            assert callback_status == LoadStatus.LOAD_FAILURE
            await self._send(State.FAILED)

    async def _handle_failure(self) -> None:
        assert self._requested_max_submit is not None

        error_msg = (
            f"Realization: {self.real.run_arg.iens} "
            f"failed after reaching max submit ({self._requested_max_submit}):"
            f"\n\t{self._callback_status_msg}"
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

    async def _send(self, state: State) -> None:
        self.state = state
        if state == State.FAILED:
            await self._handle_failure()

        elif state == State.ABORTED:
            await self._handle_aborted()

        elif state == State.COMPLETED:
            self._end_time = time.time()
            await self._scheduler.completed_jobs.put(self.iens)

        status = STATE_TO_LEGACY[state]
        event = CloudEvent(
            {
                "type": _queue_state_event_type(status),
                "source": f"/ert/ensemble/{self._scheduler._ens_id}/real/{self.iens}",
                "datacontenttype": "application/json",
            },
            {
                "queue_event_type": status,
            },
        )
        await self._scheduler._events.put(to_json(event))


def log_info_from_exit_file(exit_file_path: Path) -> None:
    if not exit_file_path.exists():
        return
    exit_file = etree.parse(exit_file_path)
    filecontents: List[str] = []
    for element in ["job", "reason", "stderr_file", "stderr"]:
        filecontents.append(str(exit_file.findtext(element)))
    logger.error(
        "job {} failed with: '{}'\n\tstderr file: '{}',\n\tits contents:{}".format(
            *filecontents
        )
    )
