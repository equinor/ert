from __future__ import annotations

import asyncio
from enum import Enum
from typing import TYPE_CHECKING

from cloudevents.conversion import to_json
from cloudevents.http import CloudEvent

from ert.callbacks import forward_model_ok
from ert.job_queue.queue import _queue_state_event_type
from ert.load_status import LoadStatus
from ert.scheduler.driver import Driver

if TYPE_CHECKING:
    from ert.ensemble_evaluator._builder._realization import Realization
    from ert.scheduler.scheduler import Scheduler


class State(str, Enum):
    WAITING = "WAITING"
    SUBMITTING = "SUBMITTING"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    ABORTING = "ABORTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ABORTED = "ABORTED"


STATE_TO_LEGACY = {
    State.WAITING: "WAITING",
    State.SUBMITTING: "SUBMITTED",
    State.STARTING: "PENDING",
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

    async def __call__(
        self, start: asyncio.Event, sem: asyncio.BoundedSemaphore
    ) -> None:
        await start.wait()
        await sem.acquire()

        try:
            await self._send(State.SUBMITTING)
            await self.driver.submit(
                self.real.iens, self.real.job_script, cwd=self.real.run_arg.runpath
            )

            await self._send(State.STARTING)
            await self.started.wait()

            await self._send(State.RUNNING)
            returncode = await self.returncode
            if (
                returncode == 0
                and forward_model_ok(self.real.run_arg).status
                == LoadStatus.LOAD_SUCCESSFUL
            ):
                await self._send(State.COMPLETED)
            else:
                await self._send(State.FAILED)

        except asyncio.CancelledError:
            await self._send(State.ABORTING)
            await self.driver.kill(self.iens)

            await self.aborted.wait()
            await self._send(State.ABORTED)
        finally:
            sem.release()

    async def _send(self, state: State) -> None:
        status = STATE_TO_LEGACY[state]
        event = CloudEvent(
            {
                "type": _queue_state_event_type(status),
                "source": f"/etc/ensemble/{self._scheduler._ens_id}/real/{self.iens}",
                "datacontenttype": "application/json",
            },
            {
                "queue_event_type": status,
            },
        )
        await self._scheduler._events.put(to_json(event))
