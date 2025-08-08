from __future__ import annotations

import asyncio
import logging
import os
import time
import traceback
from collections import defaultdict
from collections.abc import Iterable, MutableMapping, Sequence
from contextlib import suppress
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import orjson
from pydantic.dataclasses import dataclass

from _ert.events import (
    ForwardModelStepChecksum,
    RealizationEvent,
    RealizationFailed,
    RealizationStoppedLongRunning,
    SnapshotInputEvent,
)

from .driver import Driver
from .event import FinishedEvent, StartedEvent
from .job import Job, JobState

if TYPE_CHECKING:
    from ert.ensemble_evaluator import Realization

logger = logging.getLogger(__name__)


@dataclass
class _JobsJson:
    ens_id: str | None
    real_id: int
    dispatch_url: str | None
    ee_token: str | None
    experiment_id: str | None


class SubmitSleeper:
    _submit_sleep: float
    _last_started: float

    def __init__(self, submit_sleep: float) -> None:
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
        realizations: Sequence[Realization] | None = None,
        manifest_queue: asyncio.Queue[ForwardModelStepChecksum] | None = None,
        ensemble_evaluator_queue: asyncio.Queue[SnapshotInputEvent] | None = None,
        *,
        max_submit: int = 1,
        max_running: int = 1,
        submit_sleep: float = 0.0,
        ens_id: str | None = None,
    ) -> None:
        self.driver = driver
        self._ensemble_evaluator_queue = ensemble_evaluator_queue
        self._manifest_queue = manifest_queue

        self._job_tasks: MutableMapping[int, asyncio.Task[None]] = {}

        self.submit_sleep_state: SubmitSleeper | None = None
        if submit_sleep > 0:
            self.submit_sleep_state = SubmitSleeper(submit_sleep)

        self._jobs: MutableMapping[int, Job] = {
            real.iens: Job(self, real) for real in (realizations or [])
        }

        self._events: asyncio.Queue[RealizationEvent] = asyncio.Queue()
        self._running: asyncio.Event = asyncio.Event()

        self._average_job_runtime: float = 0
        self._completed_jobs_num: int = 0
        self.completed_jobs: asyncio.Queue[int] = asyncio.Queue()
        self.warnings_extracted: bool = False

        self._cancelled = False
        if max_submit < 0:
            raise ValueError(
                "max_submit needs to be a positive number. "
                "The zero value can be used internally for testing purposes only!"
            )
        self._max_submit = max_submit
        self._max_running = max_running
        self._ens_id = ens_id

        self.checksum: dict[str, dict[str, Any]] = {}

    async def kill_all_jobs(self) -> None:
        await self.cancel_all_jobs()

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

    def mark_job_as_being_killed_by_evaluator(self, real_id: int) -> None:
        """Mark the job as being killed by the evaluator"""
        self._jobs[real_id]._started_killing_by_evaluator = True

    def confirm_job_killed_by_evaluator(self, real_id: int) -> None:
        """Set the job as killed by the evaluator, if the flag for \
            evaluator started killing job is already set."""
        if self._jobs[real_id]._started_killing_by_evaluator:
            self._jobs[int(real_id)]._was_killed_by_evaluator.set()

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
                        logger.warning(
                            f"Stopping realization {iens} as its running duration "
                            f"{self._jobs[iens].running_duration}s is longer than "
                            f"the factor {long_running_factor} multiplied with the "
                            f"average runtime {self._average_job_runtime}s."
                        )
                        await self._events.put(
                            RealizationStoppedLongRunning(
                                real=str(iens), ensemble=self._ens_id
                            )
                        )
                        task.cancel()
                        with suppress(asyncio.CancelledError):
                            await task
            await asyncio.sleep(0.1)

    def set_realization(self, realization: Realization) -> None:
        self._jobs[realization.iens] = Job(self, realization)

    def is_active(self) -> bool:
        return any(not task.done() for task in self._job_tasks.values())

    def count_states(self) -> dict[JobState, int]:
        counts: dict[JobState, int] = defaultdict(int)
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

    def add_dispatch_information_to_jobs_file(
        self, ee_uri: str, ee_token: str | None
    ) -> None:
        for job in self._jobs.values():
            self._update_jobs_json(job.iens, job.real.run_arg.runpath, ee_uri, ee_token)

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
                        f"Exception in scheduler task {task.get_name()}: "
                        f"{task_exception}\n"
                        f"Traceback: {exc_traceback}"
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
    ) -> bool:
        """Run all the jobs in the scheduler, and wait for them to finish.

        Args:
            min_required_realizations (int, optional): The minimum amount of
            realizations that have to be completed before stopping
            long-running jobs. Defaults to 0.

        Returns:
            bool: Returns True if the scheduler ran successfully, False if it
            was cancelled.
        """
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
            scheduling_tasks.extend(
                (
                    asyncio.create_task(
                        self._stop_long_running_jobs(min_required_realizations)
                    ),
                    asyncio.create_task(self._update_avg_job_runtime()),
                )
            )

        sem = asyncio.BoundedSemaphore(self._max_running or len(self._jobs))
        # this lock is to assure that no more than 1 task
        # does internalization at a time
        forward_model_ok_lock = asyncio.Lock()
        verify_checksum_lock = asyncio.Lock()
        for iens, job in self._jobs.items():
            await asyncio.sleep(0)
            if job.state != JobState.ABORTED:
                self._job_tasks[iens] = asyncio.create_task(
                    job.run(
                        sem,
                        forward_model_ok_lock,
                        verify_checksum_lock,
                        self._max_submit,
                    ),
                    name=f"job-{iens}_task",
                )
            else:
                failure = job.real.run_arg.ensemble_storage.get_failure(iens)
                await self._events.put(
                    RealizationFailed(
                        ensemble=self._ens_id,
                        real=str(iens),
                        queue_event_type=JobState.FAILED,
                        message=failure.message if failure else None,
                    )
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
            return False

        return True

    async def _process_event_queue(self) -> None:
        while True:
            event = await self.driver.event_queue.get()
            job = self._jobs[event.iens]

            # Any event implies the job has at least started
            job.started.set()

            if isinstance(event, StartedEvent | FinishedEvent) and event.exec_hosts:
                self._jobs[event.iens].exec_hosts = event.exec_hosts

            if (
                isinstance(event, FinishedEvent)
                and not self._cancelled
                and not job.returncode.cancelled()
            ):
                job.returncode.set_result(event.returncode)

    def _update_jobs_json(
        self, iens: int, runpath: str, ee_uri: str, ee_token: str | None
    ) -> None:
        jobs = _JobsJson(
            experiment_id=None,
            ens_id=self._ens_id,
            real_id=iens,
            dispatch_url=ee_uri,
            ee_token=ee_token,
        )
        jobs_path = os.path.join(runpath, "jobs.json")
        try:
            with open(jobs_path, "rb") as fp:
                data = orjson.loads(fp.read())
            with open(jobs_path, "wb") as fp:
                data.update(asdict(jobs))
                fp.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))
        except OSError as err:
            error_msg = f"Could not update jobs.json: {err}"
            self._jobs[iens].unschedule(error_msg)
            logger.error(error_msg)
            return
