from __future__ import annotations

import asyncio
import logging
import shlex
from asyncio.subprocess import PIPE
from typing import List, Literal, Mapping, MutableMapping, Optional, Tuple, Union

from pydantic import BaseModel, Field
from typing_extensions import Annotated

from ert.scheduler.driver import Driver
from ert.scheduler.event import Event, FinishedEvent, StartedEvent

logger = logging.getLogger(__name__)

_POLL_PERIOD = 2.0  # seconds
JobState = Literal["B", "E", "F", "H", "M", "Q", "R", "S", "T", "U", "W", "X"]
JOBSTATE_INITIAL: JobState = "Q"


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


class OpenPBSDriver(Driver):
    """Driver targetting OpenPBS (https://github.com/openpbs/openpbs) / PBS Pro"""

    def __init__(
        self,
        *,
        queue_name: Optional[str] = None,
        memory_per_job: Optional[str] = None,
        job_prefix: Optional[str] = None,
    ) -> None:
        super().__init__()

        self._queue_name = queue_name
        self._memory_per_job = memory_per_job
        self._job_prefix = job_prefix

        self._jobs: MutableMapping[str, Tuple[int, JobState]] = {}
        self._iens2jobid: MutableMapping[int, str] = {}

    def _resource_string(self) -> str:
        resource_specifiers: List[str] = []
        if self._memory_per_job is not None:
            resource_specifiers += ["mem=" + self._memory_per_job]
        return ":".join(resource_specifiers)

    async def submit(
        self,
        iens: int,
        executable: str,
        /,
        *args: str,
        name: str = "dummy",
        runpath: Optional[str] = None,
    ) -> None:
        arg_queue_name = ["-q", self._queue_name] if self._queue_name else []
        resource_string = self._resource_string()
        arg_resource_string = ["-l", resource_string] if resource_string else []

        name_prefix = self._job_prefix or ""
        qsub_with_args: List[str] = [
            "qsub",
            "-koe",  # Discard stdout/stderr of job
            "-rn",  # Don't restart on failure
            f"-N{name_prefix}{name}",  # Set name of job
            *arg_queue_name,
            *arg_resource_string,
            "--",
            executable,
            *args,
        ]
        logger.debug(f"Submitting to PBS with command {shlex.join(qsub_with_args)}")
        process = await asyncio.create_subprocess_exec(
            *qsub_with_args,
            stdout=PIPE,
        )
        job_id, _ = await process.communicate()
        job_id_ = job_id.decode("utf-8").strip()
        logger.debug(f"Realization {iens} accepted by PBS, got id {job_id_}")
        self._jobs[job_id_] = (iens, JOBSTATE_INITIAL)
        self._iens2jobid[iens] = job_id_

    async def kill(self, iens: int) -> None:
        try:
            job_id = self._iens2jobid[iens]

            logger.debug(f"Killing realization {iens} with PBS-id {job_id}")
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
                await self._process_job_update(job_id, job)

            await asyncio.sleep(_POLL_PERIOD)

    async def _process_job_update(self, job_id: str, job: AnyJob) -> None:
        allowed_transitions = {"Q": ["R", "F"], "R": ["F"]}
        if job_id not in self._jobs:
            return

        iens, old_state = self._jobs[job_id]
        new_state = job.job_state
        if old_state == new_state:
            return
        if not new_state in allowed_transitions[old_state]:
            logger.warning(
                f"Ignoring transition from {old_state} to {new_state} in {iens=} {job_id=}"
            )
            return
        self._jobs[job_id] = (iens, new_state)
        event: Optional[Event] = None
        if isinstance(job, RunningJob):
            logger.debug(f"Realization {iens} is running")
            event = StartedEvent(iens=iens)
        elif isinstance(job, FinishedJob):
            aborted = job.returncode >= 256
            event = FinishedEvent(iens=iens, returncode=job.returncode, aborted=aborted)
            if aborted:
                logger.debug(
                    f"Realization {iens} (PBS-id: {self._iens2jobid[iens]}) failed"
                )
            else:
                logger.debug(
                    f"Realization {iens} (PBS-id: {self._iens2jobid[iens]}) succeeded"
                )
            del self._jobs[job_id]
            del self._iens2jobid[iens]

        if event:
            await self.event_queue.put(event)

    async def finish(self) -> None:
        pass
