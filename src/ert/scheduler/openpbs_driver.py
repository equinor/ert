from __future__ import annotations

import asyncio
import json
import logging
import shlex
import shutil
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast, get_type_hints

from .driver import Driver, FailedSubmit, create_submit_script
from .event import DriverEvent, FinishedEvent, StartedEvent

logger = logging.getLogger(__name__)

_POLL_PERIOD = 2.0  # seconds
JOB_STATES = [
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


@dataclass(frozen=True)
class IgnoredJobstates:
    job_state: Literal["B", "M", "S", "T", "U", "W", "X"]


@dataclass(frozen=True)
class QueuedJob:
    job_state: Literal["H", "Q"] = "H"


@dataclass(frozen=True)
class RunningJob:
    job_state: Literal["R"]


@dataclass(frozen=True)
class FinishedJob:
    job_state: Literal["E", "F"]
    returncode: int | None = None


AnyJob = FinishedJob | QueuedJob | RunningJob | IgnoredJobstates


_STATE_ORDER: dict[type[AnyJob], int] = {
    IgnoredJobstates: -1,
    QueuedJob: 0,
    RunningJob: 1,
    FinishedJob: 2,
}


def _create_job_class(job_dict: Mapping[str, str]) -> AnyJob:
    job_state = job_dict["job_state"]
    if job_state in get_type_hints(FinishedJob)["job_state"].__args__:
        return FinishedJob(
            cast(Literal["E", "F"], job_state),
            returncode=int(job_dict["Exit_status"])
            if "Exit_status" in job_dict
            else None,
        )
    if job_state in get_type_hints(RunningJob)["job_state"].__args__:
        return RunningJob("R")
    if job_state in get_type_hints(QueuedJob)["job_state"].__args__:
        return QueuedJob(cast(Literal["H", "Q"], job_state))
    if job_state in get_type_hints(IgnoredJobstates)["job_state"].__args__:
        return IgnoredJobstates(
            cast(Literal["B", "M", "S", "T", "U", "W", "X"], job_state)
        )
    raise TypeError(f"Invalid job state '{job_state}'")


def _parse_jobs_dict(jobs: Mapping[str, Mapping[str, str]]) -> dict[str, AnyJob]:
    parsed_jobs_dict: dict[str, AnyJob] = {}
    for job_id, job_dict in jobs.items():
        parsed_jobs_dict[job_id] = _create_job_class(job_dict)
    return parsed_jobs_dict


def parse_qstat(qstat_output: str) -> dict[str, dict[str, str]]:
    data: dict[str, dict[str, str]] = {}
    for line in qstat_output.splitlines():
        if line.startswith(("Job id  ", "-" * 16)):
            continue
        tokens = line.split(maxsplit=6)
        if len(tokens) >= 5 and tokens[0] and tokens[5]:
            if tokens[4] not in JOB_STATES:
                logger.error(
                    f"Unknown state {tokens[4]} obtained from "
                    f"PBS for jobid {tokens[0]}, ignored."
                )
                continue
            data[tokens[0]] = {"job_state": tokens[4]}
    return data


class OpenPBSDriver(Driver):
    """Driver targetting OpenPBS (https://github.com/openpbs/openpbs) / PBS Pro"""

    def __init__(
        self,
        *,
        queue_name: str | None = None,
        project_code: str | None = None,
        keep_qsub_output: bool | None = None,
        cluster_label: str | None = None,
        job_prefix: str | None = None,
        qsub_cmd: str | None = None,
        qstat_cmd: str | None = None,
        qdel_cmd: str | None = None,
        activate_script: str = "",
    ) -> None:
        super().__init__(activate_script)

        self._queue_name = queue_name
        self._project_code = project_code
        self._keep_qsub_output = keep_qsub_output
        self._cluster_label: str | None = cluster_label
        self._job_prefix = job_prefix
        self._max_pbs_cmd_attempts = 10
        self._sleep_time_between_cmd_retries = 2
        self._poll_period = _POLL_PERIOD

        self._qsub_cmd = Path(qsub_cmd or shutil.which("qsub") or "qsub")
        self._qstat_cmd = Path(qstat_cmd or shutil.which("qstat") or "qstat")
        self._qdel_cmd = Path(qdel_cmd or shutil.which("qdel") or "qdel")

        self._jobs: MutableMapping[str, tuple[int, AnyJob]] = {}
        self._iens2jobid: MutableMapping[int, str] = {}
        self._non_finished_job_ids: set[str] = set()
        self._finished_job_ids: set[str] = set()
        self._finished_iens: set[int] = set()

    def _build_resource_string(
        self, num_cpu: int = 1, realization_memory: int = 0
    ) -> list[str]:
        resource_specifiers: list[str] = []

        cpu_resources: list[str] = []
        if num_cpu > 1:
            cpu_resources += [f"ncpus={num_cpu}"]
        if realization_memory > 0:
            cpu_resources += [f"mem={realization_memory // 1024**2}mb"]
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
        name: str | None = None,
        runpath: Path | None = None,
        num_cpu: int | None = 1,
        realization_memory: int | None = 0,
    ) -> None:
        if runpath is None:
            runpath = Path.cwd()
        if name is None:
            name = Path(executable).name

        arg_queue_name = ["-q", self._queue_name] if self._queue_name else []
        arg_project_code = ["-A", self._project_code] if self._project_code else []
        arg_keep_qsub_output = (
            [] if self._keep_qsub_output else ["-o", "/dev/null", "-e", "/dev/null"]
        )

        script = create_submit_script(runpath, executable, args, self.activate_script)
        name_prefix = self._job_prefix or ""
        qsub_with_args: list[str] = [
            str(self._qsub_cmd),
            "-rn",  # Don't restart on failure
            f"-N{name_prefix}{name}",  # Set name of job
            *arg_queue_name,
            *arg_project_code,
            *arg_keep_qsub_output,
            *self._build_resource_string(
                num_cpu=num_cpu or 1, realization_memory=realization_memory or 0
            ),
        ]
        logger.debug(f"Submitting to PBS with command {shlex.join(qsub_with_args)}")
        logger.debug(f"Submit script passed on stdin: {script}")

        process_success, process_message = await self._execute_with_retry(
            qsub_with_args,
            retry_codes=(
                QSUB_INVALID_CREDENTIAL,
                QSUB_PREMATURE_END_OF_MESSAGE,
                QSUB_CONNECTION_REFUSED,
            ),
            stdin=script.encode(encoding="utf-8"),
            total_attempts=self._max_pbs_cmd_attempts,
            retry_interval=self._sleep_time_between_cmd_retries,
            driverlogger=logger,
        )
        if not process_success:
            self._job_error_message_by_iens[iens] = process_message
            raise FailedSubmit(process_message)

        job_id_ = process_message
        logger.debug(f"Realization {iens} accepted by PBS, got id {job_id_}")
        self._jobs[job_id_] = (iens, QueuedJob())
        self._iens2jobid[iens] = job_id_
        self._non_finished_job_ids.add(job_id_)

    async def kill(self, iens: int) -> None:
        if iens in self._finished_iens:
            return

        if iens not in self._iens2jobid:
            logger.warning(
                f"PBS kill failed due to missing jobid for realization {iens}. "
                "It might have already finished"
            )
            return

        job_id = self._iens2jobid[iens]

        logger.debug(f"Killing realization {iens} with PBS-id {job_id}")

        process_success, process_message = await self._execute_with_retry(
            [str(self._qdel_cmd), str(job_id)],
            retry_codes=(QDEL_REQUEST_INVALID,),
            accept_codes=(QDEL_JOB_HAS_FINISHED,),
            total_attempts=self._max_pbs_cmd_attempts,
            retry_interval=self._sleep_time_between_cmd_retries,
            driverlogger=logger,
        )
        if not process_success:
            raise RuntimeError(process_message)

    async def poll(self) -> None:
        while True:
            if not self._jobs:
                await asyncio.sleep(self._poll_period)
                continue

            if self._non_finished_job_ids:
                try:
                    process = await asyncio.create_subprocess_exec(
                        str(self._qstat_cmd),
                        "-Ex",
                        "-w",  # wide format
                        *self._non_finished_job_ids,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                except OSError as e:
                    logger.error(str(e))
                    await asyncio.sleep(self._poll_period)
                    continue
                stdout, stderr = await process.communicate()
                if process.returncode not in {0, QSTAT_UNKNOWN_JOB_ID}:
                    # Any unknown job ids will yield QSTAT_UNKNOWN_JOB_ID, but
                    # results for other job ids on stdout can be assumed valid.
                    await asyncio.sleep(self._poll_period)
                    continue
                if process.returncode == QSTAT_UNKNOWN_JOB_ID:
                    logger.debug(
                        f"qstat gave returncode {QSTAT_UNKNOWN_JOB_ID} "
                        f"with message {stderr.decode(errors='ignore')}"
                    )
                parsed_jobs = _parse_jobs_dict(
                    parse_qstat(stdout.decode(errors="ignore"))
                )
                for job_id, job in parsed_jobs.items():
                    if isinstance(job, FinishedJob):
                        self._non_finished_job_ids.remove(job_id)
                        self._finished_job_ids.add(job_id)
                    else:
                        await self._process_job_update(job_id, job)

            if self._finished_job_ids:
                process = await asyncio.create_subprocess_exec(
                    str(self._qstat_cmd),
                    "-Efx",
                    "-Fjson",
                    *self._finished_job_ids,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await process.communicate()
                if process.returncode not in {0, QSTAT_UNKNOWN_JOB_ID}:
                    # Any unknown job ids will yield QSTAT_UNKNOWN_JOB_ID, but
                    # results for other job ids on stdout can be assumed valid.
                    await asyncio.sleep(self._poll_period)
                    continue
                if process.returncode == QSTAT_UNKNOWN_JOB_ID:
                    logger.debug(
                        f"qstat gave returncode {QSTAT_UNKNOWN_JOB_ID} "
                        f"with message {stderr.decode(errors='ignore')}"
                    )
                stdout_content: dict[str, Any] = json.loads(
                    stdout.decode(errors="ignore")
                )
                parsed_jobs_dict = _parse_jobs_dict(stdout_content.get("Jobs", {}))
                for job_id, job in parsed_jobs_dict.items():
                    await self._process_job_update(job_id, job)

            await asyncio.sleep(self._poll_period)

    async def _process_job_update(self, job_id: str, new_state: AnyJob) -> None:
        if job_id not in self._jobs:
            return

        iens, old_state = self._jobs[job_id]
        if isinstance(new_state, IgnoredJobstates):
            logger.debug(
                f"Job ID '{job_id}' for {iens=} is of "
                f"unknown job state '{new_state.job_state}'"
            )
            return

        if _STATE_ORDER[type(new_state)] <= _STATE_ORDER[type(old_state)]:
            return

        self._jobs[job_id] = (iens, new_state)
        event: DriverEvent | None = None
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
                logger.info(
                    f"Realization {iens} (PBS-id: {self._iens2jobid[iens]}) failed"
                )
            else:
                logger.info(
                    f"Realization {iens} (PBS-id: {self._iens2jobid[iens]}) succeeded"
                )
            self._finished_iens.add(iens)
            del self._jobs[job_id]
            del self._iens2jobid[iens]
            self._finished_job_ids.remove(job_id)

        if event:
            await self.event_queue.put(event)

    async def finish(self) -> None:
        pass
