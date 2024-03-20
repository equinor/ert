from __future__ import annotations

import asyncio
import json
import logging
import os
import ssl
import time
import traceback
from collections import defaultdict
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    MutableMapping,
    Optional,
    Sequence,
)

from pydantic.dataclasses import dataclass
from websockets import Headers
from websockets.client import connect

from ert.constant_filenames import CERT_FILE
from ert.job_queue.queue import EVTYPE_ENSEMBLE_CANCELLED, EVTYPE_ENSEMBLE_STOPPED
from ert.scheduler.driver import SIGNAL_OFFSET, Driver
from ert.scheduler.event import FinishedEvent
from ert.scheduler.job import Job
from ert.scheduler.job import State as JobState

if TYPE_CHECKING:
    from ert.ensemble_evaluator._builder._realization import Realization


logger = logging.getLogger(__name__)


@dataclass
class _JobsJson:
    ens_id: Optional[str]
    real_id: int
    dispatch_url: Optional[str]
    ee_token: Optional[str]
    ee_cert_path: Optional[str]
    experiment_id: Optional[str]


class SubmitSleeper:
    _submit_sleep: float
    _last_started: float

    def __init__(self, submit_sleep: float):
        self._submit_sleep = submit_sleep
        self._last_started = (
            time.time() - submit_sleep
        )  # Allow the first to start immediately

    async def sleep_until_we_can_submit(self) -> None:
        now = time.time()
        next_start_time = max(self._last_started + self._submit_sleep, now)
        self._last_started = next_start_time
        await asyncio.sleep(max(0, next_start_time - now))


class Scheduler:
    def __init__(
        self,
        driver: Driver,
        realizations: Optional[Sequence[Realization]] = None,
        *,
        max_submit: int = 1,
        max_running: int = 1,
        submit_sleep: float = 0.0,
        ens_id: Optional[str] = None,
        ee_uri: Optional[str] = None,
        ee_cert: Optional[str] = None,
        ee_token: Optional[str] = None,
    ) -> None:
        self.driver = driver
        self._job_tasks: MutableMapping[int, asyncio.Task[None]] = {}

        self.submit_sleep_state: Optional[SubmitSleeper] = None
        if submit_sleep > 0:
            self.submit_sleep_state = SubmitSleeper(submit_sleep)

        self._jobs: MutableMapping[int, Job] = {
            real.iens: Job(self, real) for real in (realizations or [])
        }

        self._events: asyncio.Queue[Any] = asyncio.Queue()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        self._average_job_runtime: float = 0
        self._completed_jobs_num: int = 0
        self.completed_jobs: asyncio.Queue[int] = asyncio.Queue()

        self._cancelled = False
        self._max_submit = max_submit
        self._max_running = max_running

        self._ee_uri = ee_uri
        self._ens_id = ens_id
        self._ee_cert = ee_cert
        self._ee_token = ee_token

    def kill_all_jobs(self) -> None:
        assert self._loop
        # Checking that the loop is running is required because everest is closing the
        # simulation context whenever an optimization simulation batch is done
        if self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self.cancel_all_jobs(), self._loop)

    async def cancel_all_jobs(self) -> None:
        self._cancelled = True
        await self._cancel_job_tasks()

    async def _cancel_job_tasks(self) -> None:
        for task in self._job_tasks.values():
            task.cancel()
        with suppress(asyncio.TimeoutError):
            await asyncio.wait(
                self._job_tasks.values(),
                timeout=1.0,
                return_when=asyncio.ALL_COMPLETED,
            )

    async def _update_avg_job_runtime(self) -> None:
        while True:
            iens = await self.completed_jobs.get()
            self._average_job_runtime = (
                self._average_job_runtime * self._completed_jobs_num
                + self._jobs[iens].running_duration
            ) / (self._completed_jobs_num + 1)
            self._completed_jobs_num += 1

    async def _stop_long_running_jobs(
        self, minimum_required_realizations: int, long_running_factor: float = 1.25
    ) -> None:
        while True:
            if self._completed_jobs_num >= minimum_required_realizations:
                for iens, task in self._job_tasks.items():
                    if (
                        self._jobs[iens].running_duration
                        > long_running_factor * self._average_job_runtime
                        and not task.done()
                    ):
                        task.cancel()
                        await task
            await asyncio.sleep(0.1)

    def set_realization(self, realization: Realization) -> None:
        self._jobs[realization.iens] = Job(self, realization)

    def is_active(self) -> bool:
        return any(not task.done() for task in self._job_tasks.values())

    def count_states(self) -> Dict[JobState, int]:
        counts: Dict[JobState, int] = defaultdict(int)
        for job in self._jobs.values():
            counts[job.state] += 1
        return counts

    async def _publisher(self) -> None:
        if not self._ee_uri:
            return
        tls: Optional[ssl.SSLContext] = None
        if self._ee_cert:
            tls = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            tls.load_verify_locations(cadata=self._ee_cert)
        headers = Headers()
        if self._ee_token:
            headers["token"] = self._ee_token

        async for conn in connect(
            self._ee_uri,
            ssl=tls,
            extra_headers=headers,
            open_timeout=60,
            ping_timeout=60,
            ping_interval=60,
            close_timeout=60,
        ):
            while True:
                event = await self._events.get()
                await conn.send(event)

    def add_dispatch_information_to_jobs_file(self) -> None:
        for job in self._jobs.values():
            self._update_jobs_json(job.iens, job.real.run_arg.runpath)

    async def _monitor_and_handle_tasks(
        self, scheduling_tasks: list[asyncio.Task[None]]
    ) -> None:
        pending: Iterable[asyncio.Task[None]] = (
            list(self._job_tasks.values()) + scheduling_tasks
        )

        while True:
            done, pending = await asyncio.wait(
                pending,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                if task.cancelled():
                    continue
                if task_exception := task.exception():
                    exc_traceback = "".join(
                        traceback.format_exception(
                            None, task_exception, task_exception.__traceback__
                        )
                    )
                    logger.error(
                        (
                            f"Exception in scheduler task: {task_exception}\n"
                            f"Traceback: {exc_traceback}"
                        )
                    )
                    if task in scheduling_tasks:
                        await self._cancel_job_tasks()
                        raise task_exception
            if not self.is_active():
                for task in self._job_tasks.values():
                    if task.cancelled():
                        continue
                    if task_exception := task.exception():
                        raise task_exception
                return

    async def execute(
        self,
        min_required_realizations: int = 0,
    ) -> str:
        # We need to store the loop due to when calling
        # cancel jobs from another thread
        self._loop = asyncio.get_running_loop()
        scheduling_tasks = [
            asyncio.create_task(self._publisher()),
            asyncio.create_task(self._process_event_queue()),
            asyncio.create_task(self.driver.poll()),
        ]

        if min_required_realizations > 0:
            scheduling_tasks.append(
                asyncio.create_task(
                    self._stop_long_running_jobs(min_required_realizations)
                )
            )
            scheduling_tasks.append(asyncio.create_task(self._update_avg_job_runtime()))

        start = asyncio.Event()
        sem = asyncio.BoundedSemaphore(self._max_running or len(self._jobs))
        for iens, job in self._jobs.items():
            self._job_tasks[iens] = asyncio.create_task(
                job(start, sem, self._max_submit)
            )

        start.set()

        try:
            await self._monitor_and_handle_tasks(scheduling_tasks)
        finally:
            for scheduling_task in scheduling_tasks:
                scheduling_task.cancel()
        await self.driver.finish()

        if self._cancelled:
            logger.debug("scheduler cancelled, stopping jobs...")
            return EVTYPE_ENSEMBLE_CANCELLED

        return EVTYPE_ENSEMBLE_STOPPED

    async def _process_event_queue(self) -> None:
        while True:
            event = await self.driver.event_queue.get()
            job = self._jobs[event.iens]

            # Any event implies the job has at least started
            job.started.set()

            if isinstance(event, FinishedEvent):
                if event.returncode >= SIGNAL_OFFSET:
                    job.returncode.cancel()
                else:
                    job.returncode.set_result(event.returncode)

    def _update_jobs_json(self, iens: int, runpath: str) -> None:
        cert_path = f"{runpath}/{CERT_FILE}"
        if self._ee_cert is not None:
            Path(cert_path).write_text(self._ee_cert, encoding="utf-8")
        jobs = _JobsJson(
            experiment_id=None,
            ens_id=self._ens_id,
            real_id=iens,
            dispatch_url=self._ee_uri,
            ee_token=self._ee_token,
            ee_cert_path=cert_path if self._ee_cert is not None else None,
        )
        jobs_path = os.path.join(runpath, "jobs.json")
        with open(jobs_path, "r") as fp:
            data = json.load(fp)
        with open(jobs_path, "w") as fp:
            data.update(asdict(jobs))
            json.dump(data, fp, indent=4)
