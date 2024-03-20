from __future__ import annotations

import asyncio
import logging
import shlex
from pathlib import Path
from typing import (
    Dict,
    List,
    Literal,
    Mapping,
    MutableMapping,
    Optional,
    Set,
    Tuple,
    Union,
    get_args,
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
    "H",  # Held
    "M",  # Moved to another server
    "Q",  # Queued
    "R",  # Running
    "S",  # Suspended
    "T",  # Transiting
    "U",  # User suspended
    "W",  # Waiting
    "X",  # Expired (subjobs only)
]

QSUB_INVALID_CREDENTIAL = 171
QSUB_PREMATURE_END_OF_MESSAGE = 183
QSUB_CONNECTION_REFUSED = 162
QDEL_JOB_HAS_FINISHED = 35
QDEL_REQUEST_INVALID = 168
QSTAT_UNKNOWN_JOB_ID = 153


class IgnoredJobstates(BaseModel):
    job_state: Literal["B", "M", "S", "T", "U", "W", "X"]


class QueuedJob(BaseModel):
    job_state: Literal["H", "Q"] = "H"


class RunningJob(BaseModel):
    job_state: Literal["R"]


class FinishedJob(BaseModel):
    job_state: Literal["E", "F"]
    returncode: Annotated[Optional[int], Field(alias="Exit_status")] = None


AnyJob = Annotated[
    Union[FinishedJob, QueuedJob, RunningJob, IgnoredJobstates],
    Field(discriminator="job_state"),
]


_STATE_ORDER: dict[type[BaseModel], int] = {
    IgnoredJobstates: -1,
    QueuedJob: 0,
    RunningJob: 1,
    FinishedJob: 2,
}


class _Stat(BaseModel):
    jobs: Annotated[Mapping[str, AnyJob], Field(alias="Jobs")]


def parse_qstat(qstat_output: str) -> Dict[str, Dict[str, Dict[str, str]]]:
    data: Dict[str, Dict[str, str]] = {}
    for line in qstat_output.splitlines():
        if line.startswith("Job id  ") or line.startswith("-" * 16):
            continue
        tokens = line.split(maxsplit=6)
        if len(tokens) >= 5 and tokens[0] and tokens[5]:
            if tokens[4] not in get_args(JobState):
                logger.error(
                    f"Unknown state {tokens[4]} obtained from "
                    f"PBS for jobid {tokens[0]}, ignored."
                )
                continue
            data[tokens[0]] = {"job_state": tokens[4]}
    return {"Jobs": data}


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
        self._sleep_time_between_cmd_retries = 2

        self._jobs: MutableMapping[str, Tuple[int, AnyJob]] = {}
        self._iens2jobid: MutableMapping[int, str] = {}
        self._non_finished_job_ids: Set[str] = set()
        self._finished_job_ids: Set[str] = set()

        self._resources = self._resource_strings()

    def _resource_strings(self) -> List[str]:
        resource_specifiers: List[str] = []

        cpu_resources: List[str] = []
        if self._num_nodes is not None:
            cpu_resources += [f"select={self._num_nodes}"]
        if self._num_cpus_per_node is not None:
            cpu_resources += [f"ncpus={self._num_cpus_per_node}"]
        if self._memory_per_job is not None:
            cpu_resources += [f"mem={self._memory_per_job}"]
        if cpu_resources:
            resource_specifiers.append(":".join(cpu_resources))

        if self._cluster_label is not None:
            resource_specifiers += [f"{self._cluster_label}"]

        cli_args = []
        for resource_string in resource_specifiers:
            cli_args.extend(["-l", resource_string])
        return cli_args

    async def submit(
        self,
        iens: int,
        executable: str,
        /,
        *args: str,
        name: str = "dummy",
        runpath: Optional[Path] = None,
    ) -> None:
        if runpath is None:
            runpath = Path.cwd()

        arg_queue_name = ["-q", self._queue_name] if self._queue_name else []
        arg_keep_qsub_output = (
            [] if self._keep_qsub_output else "-o /dev/null -e /dev/null".split()
        )

        script = (
            "#!/usr/bin/env bash\n"
            f"cd {shlex.quote(str(runpath))}\n"
            f"exec -a {shlex.quote(executable)} {executable} {shlex.join(args)}\n"
        )
        name_prefix = self._job_prefix or ""
        qsub_with_args: List[str] = [
            "qsub",
            "-rn",  # Don't restart on failure
            f"-N{name_prefix}{name}",  # Set name of job
            *arg_queue_name,
            *arg_keep_qsub_output,
            *self._resources,
        ]
        logger.debug(f"Submitting to PBS with command {shlex.join(qsub_with_args)}")

        process_success, process_message = await self._execute_with_retry(
            qsub_with_args,
            retry_codes=(
                QSUB_INVALID_CREDENTIAL,
                QSUB_PREMATURE_END_OF_MESSAGE,
                QSUB_CONNECTION_REFUSED,
            ),
            stdin=script.encode(encoding="utf-8"),
            retries=self._num_pbs_cmd_retries,
            retry_interval=self._sleep_time_between_cmd_retries,
            driverlogger=logger,
        )
        if not process_success:
            raise RuntimeError(process_message)

        job_id_ = process_message
        logger.debug(f"Realization {iens} accepted by PBS, got id {job_id_}")
        self._jobs[job_id_] = (iens, QueuedJob())
        self._iens2jobid[iens] = job_id_
        self._non_finished_job_ids.add(job_id_)

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
            retries=self._num_pbs_cmd_retries,
            retry_interval=self._sleep_time_between_cmd_retries,
            driverlogger=logger,
        )
        if not process_success:
            raise RuntimeError(process_message)

    async def poll(self) -> None:
        while True:
            if not self._jobs:
                await asyncio.sleep(_POLL_PERIOD)
                continue

            if self._non_finished_job_ids:
                process = await asyncio.create_subprocess_exec(
                    "qstat",
                    "-x",
                    "-w",  # wide format
                    *self._non_finished_job_ids,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await process.communicate()
                if process.returncode not in {0, QSTAT_UNKNOWN_JOB_ID}:
                    # Any unknown job ids will yield QSTAT_UNKNOWN_JOB_ID, but
                    # results for other job ids on stdout can be assumed valid.
                    await asyncio.sleep(_POLL_PERIOD)
                    continue
                if process.returncode == QSTAT_UNKNOWN_JOB_ID:
                    logger.debug(
                        f"qstat gave returncode {QSTAT_UNKNOWN_JOB_ID} "
                        f"with message {stderr.decode(errors='ignore')}"
                    )
                stat = _Stat(**parse_qstat(stdout.decode(errors="ignore")))
                for job_id, job in stat.jobs.items():
                    if isinstance(job, FinishedJob):
                        self._non_finished_job_ids.remove(job_id)
                        self._finished_job_ids.add(job_id)
                    else:
                        await self._process_job_update(job_id, job)

            if self._finished_job_ids:
                process = await asyncio.create_subprocess_exec(
                    "qstat",
                    "-fx",
                    "-Fjson",
                    *self._finished_job_ids,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await process.communicate()
                if process.returncode not in {0, QSTAT_UNKNOWN_JOB_ID}:
                    # Any unknown job ids will yield QSTAT_UNKNOWN_JOB_ID, but
                    # results for other job ids on stdout can be assumed valid.
                    await asyncio.sleep(_POLL_PERIOD)
                    continue
                if process.returncode == QSTAT_UNKNOWN_JOB_ID:
                    logger.debug(
                        f"qstat gave returncode {QSTAT_UNKNOWN_JOB_ID} "
                        f"with message {stderr.decode(errors='ignore')}"
                    )
                stat = _Stat.model_validate_json(stdout.decode(errors="ignore"))
                for job_id, job in stat.jobs.items():
                    await self._process_job_update(job_id, job)

            await asyncio.sleep(_POLL_PERIOD)

    async def _process_job_update(self, job_id: str, new_state: AnyJob) -> None:
        if job_id not in self._jobs:
            return

        iens, old_state = self._jobs[job_id]
        if isinstance(new_state, IgnoredJobstates):
            logger.debug(
                f"Job ID '{job_id}' for {iens=} is of unknown job state '{new_state.job_state}'"
            )
            return

        if _STATE_ORDER[type(new_state)] <= _STATE_ORDER[type(old_state)]:
            return

        self._jobs[job_id] = (iens, new_state)
        event: Optional[Event] = None
        if isinstance(new_state, RunningJob):
            logger.debug(f"Realization {iens} is running")
            event = StartedEvent(iens=iens)
        elif isinstance(new_state, FinishedJob):
            assert new_state.returncode is not None
            event = FinishedEvent(
                iens=iens,
                returncode=new_state.returncode,
            )

            if new_state.returncode != 0:
                logger.debug(
                    f"Realization {iens} (PBS-id: {self._iens2jobid[iens]}) failed"
                )
            else:
                logger.debug(
                    f"Realization {iens} (PBS-id: {self._iens2jobid[iens]}) succeeded"
                )
            del self._jobs[job_id]
            del self._iens2jobid[iens]
            self._finished_job_ids.remove(job_id)

        if event:
            await self.event_queue.put(event)

    async def finish(self) -> None:
        pass
