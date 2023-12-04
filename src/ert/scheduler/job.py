from __future__ import annotations

import asyncio
import os
import sys
from enum import Enum
from typing import TYPE_CHECKING, Optional

from cloudevents.conversion import to_json
from cloudevents.http import CloudEvent

from ert.callbacks import forward_model_ok
from ert.job_queue.queue import _queue_state_event_type
from ert.load_status import LoadStatus

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
        self._scheduler = scheduler

    @property
    def iens(self) -> int:
        return self.real.iens

    async def __call__(
        self, start: asyncio.Event, sem: asyncio.BoundedSemaphore
    ) -> None:
        await start.wait()
        await sem.acquire()

        proc: Optional[asyncio.subprocess.Process] = None
        try:
            await self._send(State.SUBMITTING)

            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                self.real.job_script,
                cwd=self.real.run_arg.runpath,
                preexec_fn=os.setpgrp,
            )
            await self._send(State.STARTING)
            await self._send(State.RUNNING)
            returncode = await proc.wait()
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
            if proc:
                proc.kill()
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
