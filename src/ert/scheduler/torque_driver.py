from __future__ import annotations

import asyncio
import shlex
from asyncio.subprocess import PIPE
from typing import Literal, Mapping, MutableMapping, Optional, Tuple, Union

from pydantic import BaseModel, Field
from typing_extensions import Annotated

from ert.scheduler.driver import Driver
from ert.scheduler.event import Event, FinishedEvent, StartedEvent

_POLL_PERIOD = 2.0  # seconds
JobState = Literal["C", "E", "H", "Q", "R", "T", "W", "F"]


class FinishedJob(BaseModel):
    job_state: Literal["F"]
    returncode: Annotated[int, Field(alias="Exit_status")]


class QueuedJob(BaseModel):
    job_state: Literal["H", "Q"]


class RunningJob(BaseModel):
    job_state: Literal["R"]


AnyJob = Annotated[
    Union[FinishedJob, QueuedJob, RunningJob], Field(discriminator="job_state")
]


class _Stat(BaseModel):
    jobs: Annotated[Mapping[str, AnyJob], Field(alias="Jobs")]


class TorqueDriver(Driver):
    def __init__(self, *, queue_name: Optional[str] = None) -> None:
        super().__init__()

        self._queue_name = queue_name
        self._jobs: MutableMapping[str, Tuple[int, JobState]] = {}
        self._iens2jobid: MutableMapping[int, str] = {}

    async def submit(
        self, iens: int, executable: str, /, *args: str, cwd: str, name: str = "dummy"
    ) -> None:
        script = (
            "#!/usr/bin/env bash\n"
            f"cd {shlex.quote(cwd)}\n"
            f"exec -a {shlex.quote(executable)} {shlex.quote(executable)} {shlex.join(args)}\n"
        )

        arg_queue_name = ["-q", self._queue_name] if self._queue_name else []

        process = await asyncio.create_subprocess_exec(
            "qsub",
            "-koe",  # Discard stdout/stderr of job
            "-rn",  # Don't restart on failure
            f"-N{name}",  # Set name of job
            *arg_queue_name,
            "-",
            stdin=PIPE,
            stdout=PIPE,
        )
        job_id, _ = await process.communicate(script.encode())
        job_id_ = job_id.decode("utf-8").strip()
        self._jobs[job_id_] = (iens, "Q")
        self._iens2jobid[iens] = job_id_

    async def kill(self, iens: int) -> None:
        try:
            job_id = self._iens2jobid[iens]

            proc = await asyncio.create_subprocess_exec("qdel", job_id)
            await proc.wait()
        except KeyError:
            return

    async def poll(self) -> None:
        while True:
            if not self._jobs.keys():
                await asyncio.sleep(_POLL_PERIOD)
                continue

            proc = await asyncio.create_subprocess_exec(
                "qstat",
                "-fxFjson",
                *self._jobs.keys(),
                stdout=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            stat = _Stat.model_validate_json(stdout)

            for job_id, job in stat.jobs.items():
                if job_id not in self._jobs:
                    continue

                iens, old_state = self._jobs[job_id]
                new_state = job.job_state
                if old_state == new_state:
                    continue

                self._jobs[job_id] = (iens, new_state)
                event: Optional[Event] = None
                if isinstance(job, RunningJob):
                    event = StartedEvent(iens=iens)
                elif isinstance(job, FinishedJob):
                    event = FinishedEvent(
                        iens=iens,
                        returncode=job.returncode,
                        aborted=job.returncode >= 256,
                    )

                    del self._jobs[job_id]
                    del self._iens2jobid[iens]

                if event:
                    await self.event_queue.put(event)

            await asyncio.sleep(_POLL_PERIOD)

    async def finish(self) -> None:
        pass
