from __future__ import annotations

import asyncio
import logging
import os
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
    Union,
)

import orjson
from pydantic.dataclasses import dataclass

from _ert.async_utils import get_running_loop
from _ert.events import Event, ForwardModelStepChecksum, Id
from ert.constant_filenames import CERT_FILE

from .driver import Driver
from .event import FinishedEvent
from .job import Job, JobState

if TYPE_CHECKING:
    from ert.ensemble_evaluator import Realization

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
        manifest_queue: Optional[asyncio.Queue[Event]] = None,
        ensemble_evaluator_queue: Optional[asyncio.Queue[Event]] = None,
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
        self._ensemble_evaluator_queue = ensemble_evaluator_queue
        self._manifest_queue = manifest_queue

        self._job_tasks: MutableMapping[int, asyncio.Task[None]] = {}

        self.submit_sleep_state: Optional[SubmitSleeper] = None
        if submit_sleep > 0:
            self.submit_sleep_state = SubmitSleeper(submit_sleep)

        self._jobs: MutableMapping[int, Job] = {
            real.iens: Job(self, real) for real in (realizations or [])
        }

        self._loop = get_running_loop()
        self._events: asyncio.Queue[Any] = asyncio.Queue()
        self._running: asyncio.Event = asyncio.Event()

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

        self.checksum: Dict[str, Dict[str, Any]] = {}

    def kill_all_jobs(self) -> None:
        assert self._loop
        # Checking that the loop is running is required because everest is closing the
        # simulation context whenever an optimization simulation batch is done
        if self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self.cancel_all_jobs(), self._loop)

    async def cancel_all_jobs(self) -> None:
        await self._running.wait()
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
        if self._manifest_queue is None:
            return
        while True:
            event = await self._manifest_queue.get()
            if type(event) is ForwardModelStepChecksum:
                self.checksum.update(event.checksums)
            self._manifest_queue.task_done()

    async def _publisher(self) -> None:
        if self._ensemble_evaluator_queue is None:
            return
        while True:
            event = await self._events.get()
            await self._ensemble_evaluator_queue.put(event)
            self._events.task_done()

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
                if self._ensemble_evaluator_queue is not None:
                    # if there is a consumer
                    # we wait till the event queue is processed
                    await self._events.join()
                for task in self._job_tasks.values():
                    if task.cancelled():
                        continue
                    if task_exception := task.exception():
                        raise task_exception
                return

    async def execute(
        self,
        min_required_realizations: int = 0,
    ) -> Union[Id.ENSEMBLE_SUCCEEDED_TYPE, Id.ENSEMBLE_CANCELLED_TYPE]:
        scheduling_tasks = [
            asyncio.create_task(self._publisher(), name="publisher_task"),
            asyncio.create_task(
                self._process_event_queue(), name="process_event_queue_task"
            ),
            asyncio.create_task(self.driver.poll(), name="poll_task"),
            asyncio.create_task(
                self._checksum_consumer(), name="checksum_consumer_task"
            ),
        ]

        if min_required_realizations > 0:
            scheduling_tasks.append(
                asyncio.create_task(
                    self._stop_long_running_jobs(min_required_realizations)
                )
            )
            scheduling_tasks.append(asyncio.create_task(self._update_avg_job_runtime()))

        sem = asyncio.BoundedSemaphore(self._max_running or len(self._jobs))
        # this lock is to assure that no more than 1 task
        # does internalization at a time
        forward_model_ok_lock = asyncio.Lock()
        for iens, job in self._jobs.items():
            self._job_tasks[iens] = asyncio.create_task(
                job.run(sem, forward_model_ok_lock, self._max_submit),
                name=f"job-{iens}_task",
            )
        logger.info("All tasks started")
        self._running.set()
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
            return Id.ENSEMBLE_CANCELLED

        return Id.ENSEMBLE_SUCCEEDED

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
        with open(jobs_path, "rb") as fp:
            data = orjson.loads(fp.read())
        with open(jobs_path, "wb") as fp:
            data.update(asdict(jobs))
            fp.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))
