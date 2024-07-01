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

from aiohttp import ClientError
from cloudevents.exceptions import DataUnmarshallerError
from cloudevents.http import from_json
from pydantic.dataclasses import dataclass
from websockets import ConnectionClosed, Headers
from websockets.client import connect

from _ert.async_utils import get_running_loop
from ert.constant_filenames import CERT_FILE
from ert.event_type_constants import (
    EVTYPE_ENSEMBLE_CANCELLED,
    EVTYPE_ENSEMBLE_STOPPED,
    EVTYPE_FORWARD_MODEL_CHECKSUM,
)
from ert.serialization import evaluator_unmarshaller

from .driver import Driver
from .event import FinishedEvent
from .job import Job
from .job import State as JobState

if TYPE_CHECKING:
    from ert.ensemble_evaluator import Realization

logger = logging.getLogger(__name__)

CLOSE_PUBLISHER_SENTINEL = object()


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

        self._loop = get_running_loop()
        self._events: asyncio.Queue[Any] = asyncio.Queue()

        self._average_job_runtime: float = 0
        self._completed_jobs_num: int = 0
        self.completed_jobs: asyncio.Queue[int] = asyncio.Queue()

        self._cancelled = False
        if max_submit < 0:
            raise ValueError(
                "max_submit needs to be a positive number. The zero value can be used internally for testing purposes only!"
            )
        self._max_submit = max_submit
        self._max_running = max_running

        self._ee_uri = ee_uri
        self._ens_id = ens_id
        self._ee_cert = ee_cert
        self._ee_token = ee_token
        self._publisher_done = asyncio.Event()
        self._consumer_started = asyncio.Event()
        self.checksum: Dict[str, Dict[str, Any]] = {}
        self.checksum_listener: Optional[asyncio.Task[None]] = None

    async def start_manifest_listener(self) -> Optional[asyncio.Task[None]]:
        if self._ee_uri is None or "dispatch" not in self._ee_uri:
            return None

        self.checksum_listener = asyncio.create_task(
            self._checksum_consumer(), name="consumer_task"
        )
        await self._consumer_started.wait()
        return self.checksum_listener

    def wait_for_checksum(self) -> bool:
        return self._consumer_started.is_set()

    def kill_all_jobs(self) -> None:
        assert self._loop
        # Checking that the loop is running is required because everest is closing the
        # simulation context whenever an optimization simulation batch is done
        if self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self.cancel_all_jobs(), self._loop)

    async def cancel_all_jobs(self) -> None:
        self._cancelled = True
        logger.info("Cancelling all jobs")
        await self._cancel_job_tasks()

    async def _cancel_job_tasks(self) -> None:
        for task in self._job_tasks.values():
            if not task.done():
                task.cancel()
        _, pending = await asyncio.wait(
            self._job_tasks.values(),
            timeout=30.0,
            return_when=asyncio.ALL_COMPLETED,
        )
        for task in pending:
            logger.debug(f"Task {task.get_name()} was not killed properly!")

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
                        logger.info(
                            f"Stopping realization {iens} as its running duration "
                            f"{self._jobs[iens].running_duration}s is longer than "
                            f"the factor {long_running_factor} multiplied with the "
                            f"average runtime {self._average_job_runtime}s."
                        )
                        task.cancel()
                        with suppress(asyncio.CancelledError):
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

    async def _checksum_consumer(self) -> None:
        if not self._ee_uri:
            return
        tls: Optional[ssl.SSLContext] = None
        if self._ee_cert:
            tls = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            tls.load_verify_locations(cadata=self._ee_cert)
        headers = Headers()
        if self._ee_token:
            headers["token"] = self._ee_token
        event = None
        async for conn in connect(
            self._ee_uri.replace("dispatch", "client"),
            ssl=tls,
            extra_headers=headers,
            max_size=2**26,
            max_queue=500,
            open_timeout=5,
            ping_timeout=60,
            ping_interval=60,
            close_timeout=60,
        ):
            try:
                self._consumer_started.set()
                async for message in conn:
                    try:
                        event = from_json(
                            str(message), data_unmarshaller=evaluator_unmarshaller
                        )
                        if event["type"] == EVTYPE_FORWARD_MODEL_CHECKSUM:
                            self.checksum.update(event.data)
                    except DataUnmarshallerError:
                        logger.error(
                            "Scheduler checksum consumer received unknown message"
                        )
            except (ConnectionRefusedError, ConnectionClosed, ClientError) as exc:
                self._consumer_started.clear()
                logger.debug(
                    f"Scheduler connection to EnsembleEvaluator went down: {exc}"
                )

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
        event = None
        async for conn in connect(
            self._ee_uri,
            ssl=tls,
            extra_headers=headers,
            open_timeout=60,
            ping_timeout=60,
            ping_interval=60,
            close_timeout=60,
        ):
            try:
                while True:
                    if event is None:
                        event = await self._events.get()
                    if event == CLOSE_PUBLISHER_SENTINEL:
                        self._publisher_done.set()
                        return
                    await conn.send(event)
                    event = None
                    self._events.task_done()
            except ConnectionClosed:
                logger.debug("Connection to EnsembleEvalutor went down, reconnecting.")
                continue

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
                            f"Exception in scheduler task {task.get_name()}: {task_exception}\n"
                            f"Traceback: {exc_traceback}"
                        )
                    )
                    if task in scheduling_tasks:
                        await self._cancel_job_tasks()
                        raise task_exception

            if not self.is_active():
                if self._ee_uri is not None:
                    await self._events.put(CLOSE_PUBLISHER_SENTINEL)
                    await self._publisher_done.wait()
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
        listener_task = await self.start_manifest_listener()
        scheduling_tasks = [
            asyncio.create_task(self._publisher(), name="publisher_task"),
            asyncio.create_task(
                self._process_event_queue(), name="process_event_queue_task"
            ),
            asyncio.create_task(self.driver.poll(), name="poll_task"),
        ]
        if listener_task is not None:
            scheduling_tasks.append(listener_task)

        if min_required_realizations > 0:
            scheduling_tasks.append(
                asyncio.create_task(
                    self._stop_long_running_jobs(min_required_realizations)
                )
            )
            scheduling_tasks.append(asyncio.create_task(self._update_avg_job_runtime()))

        sem = asyncio.BoundedSemaphore(self._max_running or len(self._jobs))
        for iens, job in self._jobs.items():
            self._job_tasks[iens] = asyncio.create_task(
                job.run(sem, self._max_submit), name=f"job-{iens}_task"
            )

        try:
            await self._monitor_and_handle_tasks(scheduling_tasks)
            await self.driver.finish()
        finally:
            for scheduling_task in scheduling_tasks:
                scheduling_task.cancel()
            # We discard exceptions when cancelling the scheduling tasks
            await asyncio.gather(
                *scheduling_tasks,
                return_exceptions=True,
            )

        if self._cancelled:
            logger.debug("Scheduler has been cancelled, jobs are stopped.")
            return EVTYPE_ENSEMBLE_CANCELLED

        return EVTYPE_ENSEMBLE_STOPPED

    async def _process_event_queue(self) -> None:
        while True:
            event = await self.driver.event_queue.get()
            job = self._jobs[event.iens]

            # Any event implies the job has at least started
            job.started.set()

            if (
                isinstance(event, FinishedEvent)
                and not self._cancelled
                and not job.returncode.cancelled()
            ):
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
