from __future__ import annotations

import asyncio
import logging
import shlex
from typing import (
    Iterable,
    List,
    Literal,
    Mapping,
    MutableMapping,
    Optional,
    Tuple,
    Union,
)

from pydantic import BaseModel, Field
from typing_extensions import Annotated

from ert.scheduler.driver import Driver
from ert.scheduler.event import Event, FinishedEvent, StartedEvent

logger = logging.getLogger(__name__)

_POLL_PERIOD = 2.0  # seconds
JobState = Literal[
    "B",  # Begun
    "E",  # Exiting with or without errors
    "F",  # Finished (completed, failed or deleted)
    "H",  # Held,
    "M",  # Moved to another server
    "Q",  # Queued
    "R",  # Running
    "S",  # Suspended
    "T",  # Transiting
    "U",  # User suspended
    "W",  # Waiting
    "X",  # Expired (subjobs only)
]
JOBSTATE_INITIAL: JobState = "Q"

QSUB_INVALID_CREDENTIAL = 171
QSUB_PREMATURE_END_OF_MESSAGE = 183
QSUB_CONNECTION_REFUSED = 162
QDEL_JOB_HAS_FINISHED = 35
QDEL_REQUEST_INVALID = 168


class FinishedJob(BaseModel):
    job_state: Literal["F"]
    returncode: Annotated[int, Field(alias="Exit_status")]


class QueuedJob(BaseModel):
    job_state: Literal["H", "Q"]


class RunningJob(BaseModel):
    job_state: Literal["R"]


class IgnoredJobstates(BaseModel):
    job_state: Literal["B", "E", "M", "S", "T", "U", "W", "X"]


AnyJob = Annotated[
    Union[FinishedJob, QueuedJob, RunningJob, IgnoredJobstates],
    Field(discriminator="job_state"),
]


class _Stat(BaseModel):
    jobs: Annotated[Mapping[str, AnyJob], Field(alias="Jobs")]


class OpenPBSDriver(Driver):
    """Driver targetting OpenPBS (https://github.com/openpbs/openpbs) / PBS Pro"""

    def __init__(
        self,
        *,
        queue_name: Optional[str] = None,
        keep_qsub_output: Optional[str] = None,
        memory_per_job: Optional[str] = None,
        num_nodes: Optional[int] = None,
        num_cpus_per_node: Optional[int] = None,
        cluster_label: Optional[str] = None,
        job_prefix: Optional[str] = None,
    ) -> None:
        super().__init__()

        self._queue_name = queue_name
        self._keep_qsub_output = keep_qsub_output in ["1", "True", "TRUE", "T"]
        self._memory_per_job = memory_per_job
        self._num_nodes: Optional[int] = num_nodes
        self._num_cpus_per_node: Optional[int] = num_cpus_per_node
        self._cluster_label: Optional[str] = cluster_label
        self._job_prefix = job_prefix
        self._num_pbs_cmd_retries = 10
        self._retry_pbs_cmd_interval = 2

        self._jobs: MutableMapping[str, Tuple[int, JobState]] = {}
        self._iens2jobid: MutableMapping[int, str] = {}

    def _resource_string(self) -> str:
        resource_specifiers: List[str] = []
        if self._num_nodes is not None:
            resource_specifiers += [f"nodes={self._num_nodes}"]
        if self._num_cpus_per_node is not None:
            resource_specifiers += [f"ppn={self._num_cpus_per_node}"]
        if self._memory_per_job is not None:
            resource_specifiers += [f"mem={self._memory_per_job}"]
        if self._cluster_label is not None:
            resource_specifiers += [self._cluster_label]
        return ":".join(resource_specifiers)

    async def _execute_with_retry(
        self,
        cmd_with_args: List[str],
        retry_codes: Iterable[int] = (),
        accept_codes: Iterable[int] = (),
    ) -> Tuple[bool, str]:
        error_message: Optional[str] = None

        for _ in range(self._num_pbs_cmd_retries):
            process = await asyncio.create_subprocess_exec(
                *cmd_with_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            assert process.returncode is not None
            if process.returncode == 0:
                return True, stdout.decode(errors="ignore").strip()
            elif process.returncode in retry_codes:
                error_message = stderr.decode(errors="ignore").strip()
            elif process.returncode in accept_codes:
                return True, stderr.decode(errors="ignore").strip()
            else:
                error_message = (
                    f'Command "{shlex.join(cmd_with_args)}" failed '
                    f"with exit code {process.returncode} and error message: "
                    + stderr.decode(errors="ignore").strip()
                )
                logger.error(error_message)
                return False, error_message

            await asyncio.sleep(self._retry_pbs_cmd_interval)
        error_message = (
            f'Command "{shlex.join(cmd_with_args)}" failed after {self._num_pbs_cmd_retries} retries'
            f" with error {error_message}"
        )
        logger.error(error_message)
        return False, error_message

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
        arg_keep_qsub_output = (
            [] if self._keep_qsub_output else "-o /dev/null -e /dev/null".split()
        )
        resource_string = self._resource_string()
        arg_resource_string = ["-l", resource_string] if resource_string else []

        name_prefix = self._job_prefix or ""
        qsub_with_args: List[str] = [
            "qsub",
            "-rn",  # Don't restart on failure
            f"-N{name_prefix}{name}",  # Set name of job
            *arg_queue_name,
            *arg_keep_qsub_output,
            *arg_resource_string,
            "--",
            executable,
            *args,
        ]
        logger.debug(f"Submitting to PBS with command {shlex.join(qsub_with_args)}")

        process_success, process_message = await self._execute_with_retry(
            qsub_with_args,
            retry_codes=(
                QSUB_INVALID_CREDENTIAL,
                QSUB_PREMATURE_END_OF_MESSAGE,
                QSUB_CONNECTION_REFUSED,
            ),
        )
        if not process_success:
            raise RuntimeError(process_message)

        job_id_ = process_message
        logger.debug(f"Realization {iens} accepted by PBS, got id {job_id_}")
        self._jobs[job_id_] = (iens, JOBSTATE_INITIAL)
        self._iens2jobid[iens] = job_id_

    async def kill(self, iens: int) -> None:
        if iens not in self._iens2jobid:
            logger.error(f"PBS kill failed due to missing jobid for realization {iens}")
            return

        job_id = self._iens2jobid[iens]

        logger.debug(f"Killing realization {iens} with PBS-id {job_id}")

        process_success, process_message = await self._execute_with_retry(
            ["qdel", str(job_id)],
            retry_codes=(QDEL_REQUEST_INVALID,),
            accept_codes=(QDEL_JOB_HAS_FINISHED,),
        )
        if not process_success:
            raise RuntimeError(process_message)

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
            if proc.returncode != 0:
                await asyncio.sleep(_POLL_PERIOD)
                continue

            stat = _Stat.model_validate_json(stdout)

            for job_id, job in stat.jobs.items():
                await self._process_job_update(job_id, job)

            await asyncio.sleep(_POLL_PERIOD)

    async def _process_job_update(self, job_id: str, job: AnyJob) -> None:
        significant_transitions = {"Q": ["R", "F"], "R": ["F"]}
        muted_transitions = {"H": ["Q", "E"], "Q": ["H", "E"], "R": ["E"]}
        if job_id not in self._jobs:
            return

        iens, old_state = self._jobs[job_id]
        new_state = job.job_state
        if old_state == new_state:
            return
        if not new_state in significant_transitions[old_state]:
            if not new_state in muted_transitions[old_state]:
                logger.debug(
                    "Ignoring transition from "
                    f"{old_state} to {new_state} in {iens=} {job_id=}"
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
