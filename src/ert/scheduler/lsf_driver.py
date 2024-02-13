from __future__ import annotations

import asyncio
import json
import logging
import shlex
import shutil
from pathlib import Path
from typing import (
    Dict,
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

_POLL_PERIOD = 2.0  # seconds

logger = logging.getLogger(__name__)

JobState = Literal[
    "EXIT", "DONE", "PEND", "RUN", "ZOMBI", "PDONE", "SSUSP", "USUSP", "UNKWN"
]


class FinishedJob(BaseModel):
    job_state: Literal["DONE", "EXIT"]


class QueuedJob(BaseModel):
    job_state: Literal["PEND"]


class RunningJob(BaseModel):
    job_state: Literal["RUN"]


AnyJob = Annotated[
    Union[FinishedJob, QueuedJob, RunningJob], Field(discriminator="job_state")
]

LSF_INFO_JSON_FILENAME = "lsf_info.json"


class _Stat(BaseModel):
    jobs: Mapping[str, AnyJob]


def parse_bjobs(bjobs_output_raw: bytes) -> Dict[str, Dict[str, Dict[str, str]]]:
    data: Dict[str, Dict[str, str]] = {}
    for line in bjobs_output_raw.decode().splitlines():
        if not line or not line[0].isdigit():
            continue
        (jobid, _, stat, _) = line.split(maxsplit=3)
        data[jobid] = {"job_state": stat}
    return {"jobs": data}


class LsfDriver(Driver):
    def __init__(
        self,
        queue_name: Optional[str] = None,
        bsub_cmd: Optional[str] = None,
        bjobs_cmd: Optional[str] = None,
        bkill_cmd: Optional[str] = None,
    ) -> None:
        super().__init__()

        self._queue_name = queue_name

        self._bsub_cmd = Path(bsub_cmd or shutil.which("bsub") or "bsub")
        self._bjobs_cmd = Path(bjobs_cmd or shutil.which("bjobs") or "bjobs")
        self._bkill_cmd = Path(bkill_cmd or shutil.which("bkill") or "bkill")

        self._jobs: MutableMapping[str, Tuple[int, JobState]] = {}
        self._iens2jobid: MutableMapping[int, str] = {}
        self._max_attempt: int = 100
        self._retry_sleep_period = 3

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

        bsub_with_args: List[str] = (
            [str(self._bsub_cmd)] + arg_queue_name + ["-J", name, executable, *args]
        )
        logger.debug(f"Submitting to LSF with command {shlex.join(bsub_with_args)}")
        process = await asyncio.create_subprocess_exec(
            *bsub_with_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        try:
            job_id = (
                stdout.decode("utf-8")
                .strip()
                .replace("<", "")
                .replace(">", "")
                .split()[1]
            )
        except IndexError as err:
            logger.error(
                f"Command \"{' '.join(bsub_with_args)}\" failed with error message: {stderr.decode()}"
            )
            raise RuntimeError from err
        logger.info(f"Realization {iens} accepted by LSF, got id {job_id}")

        if runpath is not None:
            (Path(runpath) / LSF_INFO_JSON_FILENAME).write_text(
                json.dumps({"job_id": job_id}), encoding="utf-8"
            )
        self._jobs[job_id] = (iens, "PEND")
        self._iens2jobid[iens] = job_id

    async def kill(self, iens: int) -> None:
        try:
            job_id = self._iens2jobid[iens]

            logger.info(f"Killing realization {iens} with LSF-id {job_id}")
            proc = await asyncio.create_subprocess_exec(self._bkill_cmd, job_id)
            await proc.wait()
        except KeyError:
            return

    async def poll(self) -> None:
        while True:
            if not self._jobs.keys():
                await asyncio.sleep(_POLL_PERIOD)
                continue
            proc = await asyncio.create_subprocess_exec(
                self._bjobs_cmd,
                *self._jobs.keys(),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, _ = await proc.communicate()
            stat = _Stat(**parse_bjobs(stdout))
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
                    logger.debug(f"Realization {iens} is running.")
                    event = StartedEvent(iens=iens)
                elif isinstance(job, FinishedJob):
                    aborted = job.job_state == "EXIT"
                    event = FinishedEvent(
                        iens=iens,
                        returncode=1 if job.job_state == "EXIT" else 0,
                        aborted=aborted,
                    )
                    if aborted:
                        logger.warning(
                            f"Realization {iens} (LSF-id: {self._iens2jobid[iens]}) failed."
                        )
                    else:
                        logger.info(
                            f"Realization {iens} (LSF-id: {self._iens2jobid[iens]}) succeeded"
                        )
                    del self._jobs[job_id]
                    del self._iens2jobid[iens]

                if event:
                    await self.event_queue.put(event)

            missing_in_bjobs_output = set(self._jobs) - set(stat.jobs.keys())
            if missing_in_bjobs_output:
                logger.warning(
                    f"bjobs did not give status for job_ids {missing_in_bjobs_output}"
                )
            await asyncio.sleep(_POLL_PERIOD)

    async def finish(self) -> None:
        pass
