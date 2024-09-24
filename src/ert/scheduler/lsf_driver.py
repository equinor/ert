from __future__ import annotations

import asyncio
import itertools
import json
import logging
import re
import shlex
import shutil
import stat
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Type,
    Union,
    cast,
    get_args,
)

from .driver import SIGNAL_OFFSET, Driver
from .event import Event, FinishedEvent, StartedEvent

_POLL_PERIOD = 2.0  # seconds
LSF_FAILED_JOB = SIGNAL_OFFSET + 65  # first non signal returncode
"""Return code we use when lsf reports failed jobs"""

logger = logging.getLogger(__name__)

JobState = Literal[
    "EXIT", "DONE", "PEND", "RUN", "ZOMBI", "PDONE", "SSUSP", "USUSP", "PSUSP", "UNKWN"
]


@dataclass(frozen=True)
class IgnoredJobstates:
    job_state: Literal["UNKWN"]


@dataclass(frozen=True)
class FinishedJobSuccess:
    job_state: Literal["DONE", "PDONE"]


@dataclass(frozen=True)
class FinishedJobFailure:
    job_state: Literal["EXIT", "ZOMBI"]


@dataclass(frozen=True)
class QueuedJob:
    job_state: Literal["PEND"]


@dataclass(frozen=True)
class RunningJob:
    job_state: Literal["RUN", "SSUSP", "USUSP", "PSUSP"]


_JOBSTATE_MAP = {
    "EXIT": FinishedJobFailure,
    "DONE": FinishedJobSuccess,
    "PEND": QueuedJob,
    "RUN": RunningJob,
    "ZOMBI": FinishedJobFailure,
    "PDONE": FinishedJobSuccess,
    "SSUSP": RunningJob,
    "USUSP": RunningJob,
    "PSUSP": RunningJob,
    "UNKWN": IgnoredJobstates,
}

AnyJob = Union[
    FinishedJobSuccess, FinishedJobFailure, QueuedJob, RunningJob, IgnoredJobstates
]

_STATE_ORDER: dict[Type[AnyJob], int] = {
    IgnoredJobstates: -1,
    QueuedJob: 0,
    RunningJob: 1,
    FinishedJobSuccess: 2,
    FinishedJobFailure: 2,
}

LSF_INFO_JSON_FILENAME = "lsf_info.json"
FLAKY_SSH_RETURNCODE = 255
JOB_ALREADY_FINISHED_BKILL_MSG = "Job has already finished"
BSUB_FAILURE_MESSAGES = ("Job not submitted",)


def _parse_jobs_dict(jobs: Mapping[str, JobState]) -> dict[str, AnyJob]:
    parsed_jobs_dict: dict[str, AnyJob] = {}
    for job_id, job_state in jobs.items():
        parsed_jobs_dict[job_id] = _JOBSTATE_MAP[job_state](job_state)
    return parsed_jobs_dict


@dataclass
class JobData:
    iens: int
    job_state: AnyJob
    submitted_timestamp: float


def parse_bjobs(bjobs_output: str) -> Dict[str, JobState]:
    data: Dict[str, JobState] = {}
    for line in bjobs_output.splitlines():
        tokens = line.split(sep="^")
        if len(tokens) == 2:
            job_id, job_state = tokens
            if job_state not in get_args(JobState):
                logger.error(
                    f"Unknown state {job_state} obtained from "
                    f"LSF for jobid {job_id}, ignored."
                )
                continue
            data[job_id] = cast(JobState, job_state)
    return data


def build_resource_requirement_string(
    exclude_hosts: Sequence[str],
    realization_memory: int,
    resource_requirement: str = "",
) -> str:
    """Merge Ert-supported resource requirements with arbitrary user supplied resources

    Ert support a list of hosts to exclude via its config system, and also
    a specification of memory. In addition Ert supports an arbitrary resource
    requirement string, which could itself specify the same things. If the resource
    spec for memory is specified in the argument resource_requirement, the value from
    realization_memory argument will override.

    Args:
        exclude_hosts: List of hostnames to exclude
        realization_memory: Memory amount to reserve in bytes

    Returns:
        Resource specification string to be added to -R option to bsub.

    """
    exclude_clauses = (
        [f"hname!='{host_name}'" for host_name in exclude_hosts if host_name]
        if exclude_hosts
        else []
    )

    if realization_memory:
        # Assume MB is default. Only LSF9 supports units.
        mem_string = f"mem={realization_memory // 1024**2}"
        if "rusage" in resource_requirement:
            if "mem=" in resource_requirement:
                # Modify in-place
                resource_requirement = re.sub(
                    r"mem=\d+[KMBTEZ]?[B]?",
                    mem_string,
                    resource_requirement,
                )
            else:
                # Inject mem= in front
                resource_requirement = re.sub(
                    r"rusage\[",
                    f"rusage[{mem_string},",
                    resource_requirement,
                )
        else:
            resource_requirement = (
                f"{resource_requirement} rusage[{mem_string}]".strip()
            )

    if not resource_requirement:
        return f"select[{' && '.join(exclude_clauses)}]" if exclude_clauses else ""

    selects = re.match(r".*select\[(.*?)\].*", resource_requirement)
    if selects and exclude_clauses:
        select_clauses = selects[1].split("&&")
        select_clauses = [string.strip() for string in select_clauses]
        select_clauses.extend(exclude_clauses)
        select_string = " && ".join(select_clauses)
        return resource_requirement.replace(selects[1], select_string)

    # If 'select[' is not in the resource requirement, append the exclusion string.
    if exclude_clauses:
        return f"{resource_requirement} select[{' && '.join(exclude_clauses)}]"

    return resource_requirement


def parse_bhist(bhist_output: str) -> Dict[str, Dict[str, int]]:
    data: Dict[str, Dict[str, int]] = {}
    for line in bhist_output.splitlines():
        if line.startswith("Summary of time"):
            assert "in seconds" in line
        if not line or not line[0].isdigit():
            continue
        tokens = line.split()
        try:
            # The bhist output has data in 10 columns in fixed positions,
            # with spaces possible in field 3. Since `split()` is used
            # to parse the output, we branch on the number of tokens found.
            if len(tokens) > 10:
                data[tokens[0]] = {
                    "pending_seconds": int(tokens[-7]),
                    "running_seconds": int(tokens[-5]),
                }
            elif len(tokens) >= 6 and tokens[0] and tokens[3] and tokens[5]:
                data[tokens[0]] = {
                    "pending_seconds": int(tokens[3]),
                    "running_seconds": int(tokens[5]),
                }
            else:
                logger.warning(f'bhist parser could not parse "{line}"')
        except ValueError as err:
            logger.warning(f'bhist parser could not parse "{line}", "{err}"')
            continue
    return data


def filter_job_ids_on_submission_time(
    jobs: MutableMapping[str, JobData], submitted_before: float
) -> set[str]:
    return {
        job_id
        for job_id, job_data in jobs.items()
        if submitted_before > job_data.submitted_timestamp
    }


class LsfDriver(Driver):
    def __init__(
        self,
        queue_name: Optional[str] = None,
        project_code: Optional[str] = None,
        resource_requirement: Optional[str] = None,
        exclude_hosts: Optional[str] = None,
        bsub_cmd: Optional[str] = None,
        bjobs_cmd: Optional[str] = None,
        bkill_cmd: Optional[str] = None,
        bhist_cmd: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._queue_name = queue_name
        self._project_code = project_code
        self._resource_requirement = resource_requirement
        self._exclude_hosts = [
            host.strip() for host in (exclude_hosts.split(",") if exclude_hosts else [])
        ]

        self._bsub_cmd = Path(bsub_cmd or shutil.which("bsub") or "bsub")
        self._bjobs_cmd = Path(bjobs_cmd or shutil.which("bjobs") or "bjobs")
        self._bkill_cmd = Path(bkill_cmd or shutil.which("bkill") or "bkill")

        self._jobs: MutableMapping[str, JobData] = {}
        self._iens2jobid: MutableMapping[int, str] = {}
        self._max_attempt: int = 100
        self._sleep_time_between_bkills = 30
        self._sleep_time_between_cmd_retries = 3
        self._bsub_retries = 10

        self._poll_period = _POLL_PERIOD

        self._bhist_cmd = Path(bhist_cmd or shutil.which("bhist") or "bhist")
        self._bhist_cache: Optional[Dict[str, Dict[str, int]]] = None
        self._bhist_required_cache_age: float = 4
        self._bhist_cache_timestamp: float = time.time()

        self._submit_locks: MutableMapping[int, asyncio.Lock] = {}

    async def submit(
        self,
        iens: int,
        executable: str,
        /,
        *args: str,
        name: str = "dummy",
        runpath: Optional[Path] = None,
        num_cpu: Optional[int] = 1,
        realization_memory: Optional[int] = 0,
    ) -> None:
        if runpath is None:
            runpath = Path.cwd()

        arg_queue_name = ["-q", self._queue_name] if self._queue_name else []
        arg_project_code = ["-P", self._project_code] if self._project_code else []

        script = (
            "#!/usr/bin/env bash\n"
            f"cd {shlex.quote(str(runpath))}\n"
            f"exec -a {shlex.quote(executable)} {executable} {shlex.join(args)}\n"
        )
        script_path: Optional[Path] = None
        with tempfile.NamedTemporaryFile(
            dir=runpath,
            prefix=".lsf_submit_",
            suffix=".sh",
            mode="w",
            encoding="utf-8",
            delete=False,
        ) as script_handle:
            script_handle.write(script)
            script_path = Path(script_handle.name)
        assert script_path is not None
        script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC)

        bsub_with_args: list[str] = [
            str(self._bsub_cmd),
            *arg_queue_name,
            *arg_project_code,
            "-o",
            str(runpath / (name + ".LSF-stdout")),
            "-e",
            str(runpath / (name + ".LSF-stderr")),
            "-n",
            str(num_cpu),
            *self._build_resource_requirement_arg(
                realization_memory=realization_memory or 0
            ),
            "-J",
            name,
            str(script_path),
            str(runpath),
        ]

        if iens not in self._submit_locks:
            self._submit_locks[iens] = asyncio.Lock()

        async with self._submit_locks[iens]:
            logger.debug(f"Submitting to LSF with command {shlex.join(bsub_with_args)}")
            process_success, process_message = await self._execute_with_retry(
                bsub_with_args,
                retry_on_empty_stdout=True,
                retry_codes=(FLAKY_SSH_RETURNCODE,),
                total_attempts=self._bsub_retries,
                retry_interval=self._sleep_time_between_cmd_retries,
                error_on_msgs=BSUB_FAILURE_MESSAGES,
            )
            if not process_success:
                self._job_error_message_by_iens[iens] = process_message
                raise RuntimeError(process_message)

            match = re.search("Job <([0-9]+)> is submitted to .*queue", process_message)
            if match is None:
                raise RuntimeError(
                    f"Could not understand '{process_message}' from bsub"
                )
            job_id = match[1]
            logger.info(f"Realization {iens} accepted by LSF, got id {job_id}")

            (Path(runpath) / LSF_INFO_JSON_FILENAME).write_text(
                json.dumps({"job_id": job_id}), encoding="utf-8"
            )
            self._jobs[job_id] = JobData(
                iens=iens,
                job_state=QueuedJob(job_state="PEND"),
                submitted_timestamp=time.time(),
            )
            self._iens2jobid[iens] = job_id

    async def kill(self, iens: int) -> None:
        if iens not in self._submit_locks:
            logger.error(
                f"LSF kill failed, realization {iens} has never been submitted"
            )
            return

        async with self._submit_locks[iens]:
            if iens not in self._iens2jobid:
                logger.error(
                    f"LSF kill failed, realization {iens} was not submitted properly"
                )
                return

            job_id = self._iens2jobid[iens]

            logger.debug(f"Killing realization {iens} with LSF-id {job_id}")
            bkill_with_args: List[str] = [
                str(self._bkill_cmd),
                "-s",
                "SIGTERM",
                job_id,
            ]

            _, process_message = await self._execute_with_retry(
                bkill_with_args,
                retry_codes=(FLAKY_SSH_RETURNCODE,),
                total_attempts=3,
                retry_interval=self._sleep_time_between_cmd_retries,
                return_on_msgs=(JOB_ALREADY_FINISHED_BKILL_MSG),
            )
            await asyncio.create_subprocess_shell(
                f"sleep {self._sleep_time_between_bkills}; {self._bkill_cmd} -s SIGKILL {job_id}",
                start_new_session=True,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )

            if not re.search(
                f"Job <{job_id}> is being (terminated|signaled)", process_message
            ):
                if JOB_ALREADY_FINISHED_BKILL_MSG in process_message:
                    logger.debug(f"LSF kill failed with: {process_message}")
                    return
                logger.error(f"LSF kill failed with: {process_message}")

    async def poll(self) -> None:
        while True:
            if not self._jobs.keys():
                await asyncio.sleep(self._poll_period)
                continue
            current_jobids = list(self._jobs.keys())

            try:
                process = await asyncio.create_subprocess_exec(
                    str(self._bjobs_cmd),
                    "-noheader",
                    "-o",
                    "jobid stat delimiter='^'",
                    *current_jobids,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except FileNotFoundError as e:
                logger.error(str(e))
                return

            stdout, stderr = await process.communicate()
            if process.returncode:
                # bjobs may give nonzero return code even when it is providing
                # at least some correct information
                logger.warning(
                    f"bjobs gave returncode {process.returncode} and error {stderr.decode()}"
                )
            bjobs_states = _parse_jobs_dict(parse_bjobs(stdout.decode(errors="ignore")))

            job_ids_found_in_bjobs_output = set(bjobs_states.keys())
            if (
                missing_in_bjobs_output := filter_job_ids_on_submission_time(
                    self._jobs, submitted_before=time.time() - self._poll_period
                )
                - job_ids_found_in_bjobs_output
            ):
                logger.debug(f"bhist is used for job ids: {missing_in_bjobs_output}")
                bhist_states = await self._poll_once_by_bhist(missing_in_bjobs_output)
                missing_in_bhist_and_bjobs = missing_in_bjobs_output - set(
                    bhist_states.keys()
                )
            else:
                bhist_states = {}
                missing_in_bhist_and_bjobs = set()

            for job_id, job in itertools.chain(
                bjobs_states.items(), bhist_states.items()
            ):
                await self._process_job_update(job_id, new_state=job)

            if missing_in_bhist_and_bjobs and self._bhist_cache is not None:
                logger.debug(
                    f"bhist did not give status for job_ids {missing_in_bhist_and_bjobs}, giving up for now."
                )
            await asyncio.sleep(self._poll_period)

    async def _process_job_update(self, job_id: str, new_state: AnyJob) -> None:
        if job_id not in self._jobs:
            return
        old_state = self._jobs[job_id].job_state
        iens = self._jobs[job_id].iens
        if isinstance(new_state, IgnoredJobstates):
            logger.debug(
                f"Job ID '{job_id}' for {iens=} is of unknown job state '{new_state.job_state}'"
            )
            return

        if _STATE_ORDER[type(new_state)] <= _STATE_ORDER[type(old_state)]:
            return

        self._jobs[job_id].job_state = new_state
        event: Optional[Event] = None
        if isinstance(new_state, RunningJob):
            logger.debug(f"Realization {iens} is running")
            event = StartedEvent(iens=iens)
        elif isinstance(new_state, FinishedJobFailure):
            logger.info(f"Realization {iens} (LSF-id: {self._iens2jobid[iens]}) failed")
            exit_code = await self._get_exit_code(job_id)
            event = FinishedEvent(iens=iens, returncode=exit_code)

        elif isinstance(new_state, FinishedJobSuccess):
            logger.info(
                f"Realization {iens} (LSF-id: {self._iens2jobid[iens]}) succeeded"
            )
            event = FinishedEvent(iens=iens, returncode=0)

        if event:
            if isinstance(event, FinishedEvent):
                del self._jobs[job_id]
                del self._iens2jobid[iens]
                await self._log_bhist_job_summary(job_id)
            await self.event_queue.put(event)

    async def _get_exit_code(self, job_id: str) -> int:
        success, output = await self._execute_with_retry(
            [f"{self._bjobs_cmd}", "-o exit_code", "-noheader", f"{job_id}"],
            retry_codes=(FLAKY_SSH_RETURNCODE,),
            total_attempts=3,
            retry_interval=self._sleep_time_between_cmd_retries,
        )

        if not success:
            return await self._get_exit_code_from_bhist(job_id)
        else:
            try:
                return int(output)
            except ValueError:
                # bjobs will sometimes return only "-" as exit code.
                # running bhist will not help in this case.
                return LSF_FAILED_JOB

    async def _get_exit_code_from_bhist(self, job_id: str) -> int:
        success, output = await self._execute_with_retry(
            [f"{self._bhist_cmd}", "-l", "-n2", f"{job_id}"],
            retry_codes=(FLAKY_SSH_RETURNCODE,),
            total_attempts=3,
            retry_interval=self._sleep_time_between_cmd_retries,
        )

        if success:
            matches = re.search(r"Exited with exit code ([0-9]+)", output)
            if matches is not None:
                return int(matches.group(1))

        return LSF_FAILED_JOB

    async def _log_bhist_job_summary(self, job_id: str) -> None:
        bhist_with_args: list[str] = [
            str(self._bhist_cmd),
            "-l",  # long format
            job_id,
        ]
        _, process_message = await self._execute_with_retry(
            bhist_with_args,
            retry_codes=(FLAKY_SSH_RETURNCODE,),
            total_attempts=3,
            retry_interval=self._sleep_time_between_cmd_retries,
            log_to_debug=False,
        )
        logger.info(f"Output from bhist -l: {process_message}")

    async def _poll_once_by_bhist(
        self, missing_job_ids: Iterable[str]
    ) -> Dict[str, AnyJob]:
        if time.time() - self._bhist_cache_timestamp < self._bhist_required_cache_age:
            return {}

        process = await asyncio.create_subprocess_exec(
            self._bhist_cmd,
            *[str(job_id) for job_id in missing_job_ids],
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode:
            logger.error(
                f"bhist gave returncode {process.returncode} with "
                f"output{stdout.decode(errors='ignore').strip()} "
                f"and error {stderr.decode(errors='ignore').strip()}"
            )
            return {}

        data: Dict[str, Dict[str, int]] = parse_bhist(stdout.decode())

        if not self._bhist_cache:
            # Boot-strapping. We can't give any data until we have run again.
            self._bhist_cache = data
            return {}

        jobs: dict[str, JobState] = {}
        for job_id, job_stat in data.items():
            if job_id not in self._bhist_cache:
                continue
            if (
                job_stat["pending_seconds"]
                == self._bhist_cache[job_id]["pending_seconds"]
                and job_stat["running_seconds"]
                == self._bhist_cache[job_id]["running_seconds"]
            ):
                jobs[job_id] = cast(JobState, "DONE")  # or EXIT, we can't tell
            elif (
                job_stat["running_seconds"]
                > self._bhist_cache[job_id]["running_seconds"]
            ):
                jobs[job_id] = cast(JobState, "RUN")
            elif (
                job_stat["pending_seconds"]
                > self._bhist_cache[job_id]["pending_seconds"]
            ):
                jobs[job_id] = cast(JobState, "PEND")
        self._bhist_cache = data
        self._bhist_cache_timestamp = time.time()
        return _parse_jobs_dict(jobs)

    def _build_resource_requirement_arg(self, realization_memory: int) -> List[str]:
        resource_requirement_string = build_resource_requirement_string(
            self._exclude_hosts,
            realization_memory,
            self._resource_requirement or "",
        )
        return (
            ["-R", resource_requirement_string] if resource_requirement_string else []
        )

    async def finish(self) -> None:
        pass

    def read_stdout_and_stderr_files(
        self, runpath: str, job_name: str, num_characters_to_read_from_end: int = 300
    ) -> str:
        error_msg = ""
        stderr_file = Path(runpath) / (job_name + ".LSF-stderr")
        if msg := tail_textfile(stderr_file, num_characters_to_read_from_end):
            error_msg += f"\n    LSF-stderr:\n{msg}"
        stdout_file = Path(runpath) / (job_name + ".LSF-stdout")
        if msg := tail_textfile(stdout_file, num_characters_to_read_from_end):
            error_msg += f"\n    LSF-stdout:\n{msg}"
        return error_msg


def tail_textfile(file_path: Path, num_chars: int) -> str:
    if not file_path.exists():
        return f"No output file {file_path}"
    with open(file_path, encoding="utf-8") as file:
        file.seek(0, 2)
        file_end_position = file.tell()
        seek_position = max(0, file_end_position - num_chars)
        file.seek(seek_position)
        return file.read()[-num_chars:]
