from __future__ import annotations

import asyncio
import logging
import uuid
from enum import Enum
from typing import TYPE_CHECKING, Optional

from cloudevents.conversion import to_json
from cloudevents.http import CloudEvent

from ert.callbacks import forward_model_ok
from ert.job_queue.queue import _queue_state_event_type
from ert.load_status import LoadStatus
from ert.scheduler.driver import Driver

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
        self.aborted = asyncio.Event()
        self._scheduler = scheduler

    @property
    def iens(self) -> int:
        return self.real.iens

    @property
    def driver(self) -> Driver:
        return self._scheduler.driver

    async def _submit_and_run_once(self, sem: asyncio.BoundedSemaphore) -> None:
        await sem.acquire()
        timeout_task: Optional[asyncio.Task[None]] = None

        try:
            await self._send(State.SUBMITTING)
            await self.driver.submit(
                self.real.iens, self.real.job_script, cwd=self.real.run_arg.runpath
            )

            await self._send(State.PENDING)
            await self.started.wait()

            await self._send(State.RUNNING)
            if self.real.max_runtime is not None and self.real.max_runtime > 0:
                timeout_task = asyncio.create_task(self._max_runtime_task())
            while not self.returncode.done():
                await asyncio.sleep(0.01)
            returncode = await self.returncode

            if (
                returncode == 0
                and forward_model_ok(self.real.run_arg).status
                == LoadStatus.LOAD_SUCCESSFUL
            ):
                await self._send(State.COMPLETED)
            else:
                await self._send(State.FAILED)
                self.returncode = asyncio.Future()
                self.started = asyncio.Event()

        except asyncio.CancelledError:
            await self._send(State.ABORTING)
            await self.driver.kill(self.iens)
            await self.aborted.wait()
            await self._send(State.ABORTED)
        finally:
            if timeout_task and not timeout_task.done():
                timeout_task.cancel()
            sem.release()

    async def __call__(
        self, start: asyncio.Event, sem: asyncio.BoundedSemaphore, max_submit: int = 2
    ) -> None:
        await start.wait()

        for attempt in range(max_submit):
            await self._submit_and_run_once(sem)

            if self.returncode.done() or self.aborted.is_set():
                break
            elif attempt < max_submit - 1:
                message = f"Realization: {self.iens} failed, resubmitting"
                logger.warning(message)
        else:
            message = f"Realization {self.iens} failed after {max_submit} attempt(s)"
            logger.error(message)

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

        self.returncode.cancel()  # Triggers CancelledError

    async def _send(self, state: State) -> None:
        self.state = state
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
