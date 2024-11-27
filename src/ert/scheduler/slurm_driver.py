from __future__ import annotations

import asyncio
import datetime
import itertools
import logging
import shlex
import stat
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import (
    Iterator,
    Optional,
    Tuple,
)

from .driver import SIGNAL_OFFSET, Driver, FailedSubmit, create_submit_script
from .event import Event, FinishedEvent, StartedEvent

SLURM_FAILED_EXIT_CODE_FETCH = SIGNAL_OFFSET + 66

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    PENDING = auto()
    COMPLETED = auto()
    RUNNING = auto()
    FAILED = auto()
    CANCELLED = auto()
    COMPLETING = auto()
    CONFIGURING = auto()


@dataclass
class JobData:
    iens: int
    exit_code: Optional[int] = None
    status: Optional[JobStatus] = None


END_STATES = {JobStatus.FAILED, JobStatus.COMPLETED, JobStatus.CANCELLED}


@dataclass
class JobInfo:
    status: Optional[JobStatus] = None


@dataclass
class ScontrolInfo(JobInfo):
    exit_code: Optional[int] = None


@dataclass
class SqueueInfo(JobInfo):
    pass


class SlurmDriver(Driver):
    def __init__(
        self,
        exclude_hosts: str = "",
        include_hosts: str = "",
        squeue_cmd: str = "squeue",
        scontrol_cmd: str = "scontrol",
        scancel_cmd: str = "scancel",
        sbatch_cmd: str = "sbatch",
        user: Optional[str] = None,
        memory: Optional[str] = "",
        realization_memory: Optional[int] = 0,
        queue_name: Optional[str] = None,
        memory_per_cpu: Optional[str] = None,
        max_runtime: Optional[float] = None,
        squeue_timeout: float = 2,
        project_code: Optional[str] = None,
        activate_script: str = "",
    ) -> None:
        """
        The arguments "memory" and "realization_memory" are currently both
        present, where the latter has been added later. "realization_memory" is
        a global keyword in Ert and not queue specific. It is always supplied
        in bytes to the driver. "memory" is a string with a Slurm unit and is a
        Slurm queue specific queue option. They are both supplied to sbatch
        using "--mem" so they cannot both be defined at the same time.

        In slurm, --mem==0 requests all memory on a node. In Ert,
        zero "realization memory" is the default and means no intended
        memory allocation.
        """
        super().__init__(activate_script)
        self._submit_locks: dict[int, asyncio.Lock] = {}
        self._iens2jobid: dict[int, str] = {}
        self._jobs: dict[str, JobData] = {}
        self._job_error_message_by_iens: dict[int, str] = {}
        self._memory_per_cpu = memory_per_cpu
        self._memory = memory
        self._realization_memory = realization_memory

        if self._realization_memory and self._memory:
            raise ValueError(
                "Overspecified memory, use either memory "
                "or realization_memory, not both"
            )

        self._max_runtime = max_runtime
        self._queue_name = queue_name

        self._exclude_hosts = exclude_hosts
        self._include_hosts = include_hosts

        self._sbatch = sbatch_cmd
        self._max_sbatch_attempts = 1

        self._scancel = scancel_cmd
        self._squeue = squeue_cmd

        self._scontrol = scontrol_cmd
        self._scontrol_cache_timestamp = 0.0
        self._scontrol_required_cache_age = 30
        self._scontrol_cache: dict[str, ScontrolInfo] = {}

        self._user = user

        self._sleep_time_between_cmd_retries = 3
        self._sleep_time_between_kills = 30
        self._poll_period = squeue_timeout
        self._project_code = project_code

    def _submit_cmd(
        self,
        name: str = "dummy",
        runpath: Optional[Path] = None,
        num_cpu: Optional[int] = 1,
    ) -> list[str]:
        sbatch_with_args = [
            str(self._sbatch),
            f"--job-name={name}",
            f"--chdir={runpath}",
            "--parsable",
            f"--output={name}.stdout",
            f"--error={name}.stderr",
        ]
        if num_cpu:
            sbatch_with_args.append(f"--ntasks={num_cpu}")
        if self._realization_memory and self._realization_memory > 0:
            # In slurm, --mem==0 requests all memory on a node. In Ert,
            # zero realization memory means no intended memory allocation.
            sbatch_with_args.append(f"--mem={self._realization_memory // 1024**2}M")
        if self._memory:
            sbatch_with_args.append(f"--mem={self._memory}")
        if self._include_hosts:
            sbatch_with_args.append(f"--nodelist={self._include_hosts}")
        if self._exclude_hosts:
            sbatch_with_args.append(f"--exclude={self._exclude_hosts}")
        if self._max_runtime and int(self._max_runtime):
            sbatch_with_args.append(
                f"--time={_seconds_to_slurm_time_format(self._max_runtime)}"
            )
        if self._memory_per_cpu:
            sbatch_with_args.append(f"--mem-per-cpu={self._memory_per_cpu}")
        if self._queue_name:
            sbatch_with_args.append(f"--partition={self._queue_name}")
        if self._project_code:
            sbatch_with_args.append(f"--account={self._project_code}")
        return sbatch_with_args

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

        script = create_submit_script(runpath, executable, args, self.activate_script)
        script_path: Optional[Path] = None
        try:
            with NamedTemporaryFile(
                dir=runpath,
                prefix=".slurm_submit_",
                suffix=".sh",
                mode="w",
                encoding="utf-8",
                delete=False,
            ) as script_handle:
                script_handle.write(script)
                script_path = Path(script_handle.name)
        except OSError as err:
            error_message = f"Could not create submit script: {err}"
            self._job_error_message_by_iens[iens] = error_message
            raise FailedSubmit(error_message) from err
        assert script_path is not None
        script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC)
        sbatch_with_args = [*self._submit_cmd(name, runpath, num_cpu), str(script_path)]

        if iens not in self._submit_locks:
            self._submit_locks[iens] = asyncio.Lock()

        async with self._submit_locks[iens]:
            logger.debug(
                f"Submitting to SLURM with command {shlex.join(sbatch_with_args)}"
            )
            process_success, process_message = await self._execute_with_retry(
                sbatch_with_args,
                retry_on_empty_stdout=True,
                retry_codes=(),
                total_attempts=self._max_sbatch_attempts,
                retry_interval=self._sleep_time_between_cmd_retries,
            )
            if not process_success:
                self._job_error_message_by_iens[iens] = process_message
                raise FailedSubmit(process_message)

            if not process_message:
                raise FailedSubmit("sbatch returned empty jobid")
            job_id = process_message
            logger.info(f"Realization {iens} accepted by SLURM, got id {job_id}")

            self._jobs[job_id] = JobData(
                iens=iens,
            )
            self._iens2jobid[iens] = job_id

    async def kill(self, iens: int) -> None:
        if iens not in self._submit_locks:
            logger.error(f"scancel failed, realization {iens} has never been submitted")
            return

        async with self._submit_locks[iens]:
            if iens not in self._iens2jobid:
                logger.error(
                    f"scancel failed, realization {iens} was not submitted properly"
                )
                return

            job_id = self._iens2jobid[iens]

            logger.debug(f"Killing realization {iens} with SLURM-id {job_id}")
            await self._execute_with_retry(
                [
                    self._scancel,
                    str(job_id),
                ]
            )

    async def poll(self) -> None:
        while True:
            if not self._jobs.keys():
                await asyncio.sleep(self._poll_period)
                continue
            arguments = ["-h", "--format=%i %T"]
            if self._user:
                arguments.append(f"--user={self._user}")
            try:
                process = await asyncio.create_subprocess_exec(
                    str(self._squeue),
                    *arguments,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except FileNotFoundError as e:
                logger.error(str(e))
                return
            stdout, stderr = await process.communicate()
            if process.returncode:
                logger.warning(
                    f"squeue gave returncode {process.returncode} and error {stderr.decode()}"
                )
            squeue_states = dict(_parse_squeue_output(stdout.decode(errors="ignore")))

            job_ids_found_in_squeue_output = set(squeue_states.keys())
            if missing_in_squeue_output := (
                set(self._jobs) - job_ids_found_in_squeue_output
            ):
                logger.debug(
                    f"scontrol is used for job ids: {missing_in_squeue_output}"
                )
                scontrol_states = {}
                for job_id in missing_in_squeue_output:
                    if (
                        scontrol_info := await self._poll_once_by_scontrol(job_id)
                    ) is not None:
                        scontrol_states[job_id] = scontrol_info
                missing_in_squeue_and_scontrol = missing_in_squeue_output - set(
                    scontrol_states.keys()
                )
            else:
                scontrol_states = {}
                missing_in_squeue_and_scontrol = set()

            for job_id, info in itertools.chain(
                squeue_states.items(), scontrol_states.items()
            ):
                await self._process_job_update(job_id, info)

            if missing_in_squeue_and_scontrol:
                logger.debug(
                    f"scontrol did not give status for job_ids {missing_in_squeue_and_scontrol}, giving up for now."
                )
            await asyncio.sleep(self._poll_period)

    async def _process_job_update(self, job_id: str, new_info: JobInfo) -> None:
        new_state = new_info.status

        if job_id not in self._jobs:
            return

        iens = self._jobs[job_id].iens

        old_state = self._jobs[job_id].status
        if old_state == new_state:
            return

        self._jobs[job_id].status = new_state
        event: Optional[Event] = None
        if new_state == JobStatus.RUNNING:
            logger.debug(f"Realization {iens} is running")
            event = StartedEvent(iens=iens)
        elif new_state == JobStatus.FAILED:
            logger.info(
                f"Realization {iens} (SLURM-id: {self._iens2jobid[iens]}) failed"
            )
            exit_code = await self._get_exit_code(job_id)
            event = FinishedEvent(iens=iens, returncode=exit_code)
        elif new_state in END_STATES:
            logger.info(
                f"Realization {iens} (SLURM-id: {self._iens2jobid[iens]}) succeeded"
            )
            event = FinishedEvent(iens=iens, returncode=0)

        if event:
            if isinstance(event, FinishedEvent):
                del self._jobs[job_id]
                del self._iens2jobid[iens]
            await self.event_queue.put(event)

    async def _get_exit_code(self, job_id: str) -> int:
        retries = 0
        while retries < 10 and self._jobs[job_id].exit_code is None:
            retries += 1
            if (scontrol_info := await self._poll_once_by_scontrol(job_id)) is not None:
                self._jobs[job_id].exit_code = scontrol_info.exit_code
            else:
                await asyncio.sleep(self._poll_period)

        code = self._jobs[job_id].exit_code
        if code is not None:
            return code
        return SLURM_FAILED_EXIT_CODE_FETCH

    async def _poll_once_by_scontrol(
        self, missing_job_id: str
    ) -> Optional[ScontrolInfo]:
        if (
            time.time() - self._scontrol_cache_timestamp
            < self._scontrol_required_cache_age
        ) and missing_job_id in self._scontrol_cache:
            return self._scontrol_cache[missing_job_id]

        process = await asyncio.create_subprocess_exec(
            self._scontrol,
            "show",
            "job",
            str(missing_job_id),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode:
            logger.error(
                f"scontrol gave returncode {process.returncode} with "
                f"output{stdout.decode(errors='ignore').strip()} "
                f"and error {stderr.decode(errors='ignore').strip()}"
            )
            return None

        info = None
        try:
            info = _parse_scontrol_output(stdout.decode(errors="ignore"))
        except Exception as err:
            logger.error(
                f"Could no parse scontrol stdout {stdout.decode(errors='ignore')}: {err}"
            )
            return info
        self._scontrol_cache[missing_job_id] = info
        self._scontrol_cache_timestamp = time.time()
        return info

    async def finish(self) -> None:
        pass

    def read_stdout_and_stderr_files(
        self, runpath: str, job_name: str, num_characters_to_read_from_end: int = 300
    ) -> str:
        error_msg = ""
        stderr_file = Path(runpath) / (job_name + ".stderr")
        if msg := _tail_textfile(stderr_file, num_characters_to_read_from_end):
            error_msg += f"\n    stderr:\n{msg}"
        stdout_file = Path(runpath) / (job_name + ".stdout")
        if msg := _tail_textfile(stdout_file, num_characters_to_read_from_end):
            error_msg += f"\n    stdout:\n{msg}"
        return error_msg


def _tail_textfile(file_path: Path, num_chars: int) -> str:
    if not file_path.exists():
        return f"No output file {file_path}"
    with open(file_path, encoding="utf-8") as file:
        file.seek(0, 2)
        file_end_position = file.tell()
        seek_position = max(0, file_end_position - num_chars)
        file.seek(seek_position)
        return file.read()[-num_chars:]


def _parse_squeue_output(output: str) -> Iterator[Tuple[str, SqueueInfo]]:
    for line in output.split("\n"):
        if line:
            id, status = line.split()
            yield id, SqueueInfo(JobStatus[status])


def _seconds_to_slurm_time_format(seconds: float) -> str:
    days = datetime.timedelta(seconds=int(seconds)).days
    hhmmss = str(
        datetime.timedelta(seconds=int(seconds)) - datetime.timedelta(days=days)
    )
    if days:
        return f"{days}-{hhmmss}"
    return hhmmss


def _parse_scontrol_output(output: str) -> ScontrolInfo:
    values = dict(w.split("=", 1) for w in output.split())
    exit_code_str = values.get("ExitCode")
    exit_code = None
    if exit_code_str:
        exit_code = int(exit_code_str.split(":")[0])
    return ScontrolInfo(JobStatus[values["JobState"]], exit_code)
