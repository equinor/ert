"""
Module implementing a queue for managing external jobs.

"""

from __future__ import annotations

import asyncio
import json
import logging
import ssl
from collections import deque
from threading import BoundedSemaphore, Semaphore
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Sequence, Tuple, Union

from cloudevents.conversion import to_json
from cloudevents.http import CloudEvent
from cwrap import BaseCClass
from websockets.client import WebSocketClientProtocol, connect
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed

from ert.config import QueueConfig
from ert.constant_filenames import CERT_FILE, JOBS_FILE
from ert.event_type_constants import (
    EVTYPE_ENSEMBLE_CANCELLED,
    EVTYPE_ENSEMBLE_FAILED,
    EVTYPE_ENSEMBLE_STOPPED,
    EVTYPE_REALIZATION_FAILURE,
    EVTYPE_REALIZATION_PENDING,
    EVTYPE_REALIZATION_RUNNING,
    EVTYPE_REALIZATION_SUCCESS,
    EVTYPE_REALIZATION_UNKNOWN,
    EVTYPE_REALIZATION_WAITING,
)
from ert.job_queue.job_queue_node import JobQueueNode
from ert.job_queue.job_status import JobStatus
from ert.job_queue.queue_differ import QueueDiffer
from ert.job_queue.thread_status import ThreadStatus

from . import ResPrototype
from .driver import Driver

if TYPE_CHECKING:
    from ert.ensemble_evaluator import Realization
    from ert.run_arg import RunArg


logger = logging.getLogger(__name__)

CLOSE_PUBLISHER_SENTINEL = object()
LONG_RUNNING_FACTOR = 1.25
"""If STOP_LONG_RUNNING is true, realizations taking more time than the average
times this ￼factor will be killed."""
CONCURRENT_INTERNALIZATION = 1
"""How many realizations allowed to be concurrently internalized using
threads."""


_queue_state_to_event_type_map = {
    "NOT_ACTIVE": EVTYPE_REALIZATION_WAITING,
    "WAITING": EVTYPE_REALIZATION_WAITING,
    "SUBMITTED": EVTYPE_REALIZATION_WAITING,
    "PENDING": EVTYPE_REALIZATION_PENDING,
    "RUNNING": EVTYPE_REALIZATION_RUNNING,
    "DONE": EVTYPE_REALIZATION_RUNNING,
    "EXIT": EVTYPE_REALIZATION_RUNNING,
    "IS_KILLED": EVTYPE_REALIZATION_FAILURE,
    "DO_KILL": EVTYPE_REALIZATION_FAILURE,
    "SUCCESS": EVTYPE_REALIZATION_SUCCESS,
    "STATUS_FAILURE": EVTYPE_REALIZATION_UNKNOWN,
    "FAILED": EVTYPE_REALIZATION_FAILURE,
    "DO_KILL_NODE_FAILURE": EVTYPE_REALIZATION_FAILURE,
    "UNKNOWN": EVTYPE_REALIZATION_UNKNOWN,
}


def _queue_state_event_type(state: str) -> str:
    return _queue_state_to_event_type_map[state]


class JobQueue(BaseCClass):  # type: ignore
    TYPE_NAME = "job_queue"
    _alloc = ResPrototype("void* job_queue_alloc(void*)", bind=False)
    _free = ResPrototype("void job_queue_free( job_queue )")
    _add_job = ResPrototype("int job_queue_add_job_node(job_queue, job_queue_node)")

    def __repr__(self) -> str:
        return f"JobQueue({self.driver}, {self.max_submit})"

    def __str__(self) -> str:
        return self.__repr__()

    def __init__(
        self,
        queue_config: QueueConfig,
        realizations: Optional[Sequence[Realization]] = None,
        *,
        ens_id: Optional[str] = None,
        ee_uri: Optional[str] = None,
        ee_cert: Optional[str] = None,
        ee_token: Optional[str] = None,
        on_timeout: Optional[Callable[[int], None]] = None,
        verify_token: bool = True,
    ) -> None:
        self.job_list: List[JobQueueNode] = []
        self._stopped = False
        self.driver: Driver = Driver.create_driver(queue_config)
        c_ptr = self._alloc(self.driver.from_param(self.driver))
        super().__init__(c_ptr)

        self._differ = QueueDiffer()
        self._max_submit = queue_config.max_submit
        self._pool_sema = BoundedSemaphore(value=CONCURRENT_INTERNALIZATION)
        self._on_timeout = on_timeout

        self._ens_id = ens_id
        self._ee_uri = ee_uri
        self._ee_cert = ee_cert
        self._ee_token = ee_token

        self._ee_ssl_context: Optional[Union[ssl.SSLContext, bool]] = None
        if ee_cert is not None:
            self._ee_ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            if verify_token:
                self._ee_ssl_context.load_verify_locations(cadata=ee_cert)
        else:
            self._ee_ssl_context = True if ee_uri and ee_uri.startswith("wss") else None

        self._changes_to_publish: Optional[
            asyncio.Queue[Union[Dict[int, str], object]]
        ] = None

        for real in realizations or []:
            self.add_realization(real)

    def get_max_running(self) -> int:
        return self.driver.get_max_running()

    def set_max_running(self, max_running: int) -> None:
        self.driver.set_max_running(max_running)

    @property
    def max_submit(self) -> int:
        return self._max_submit

    def free(self) -> None:
        self._free()

    def is_active(self) -> bool:
        return any(
            job.thread_status
            in (ThreadStatus.READY, ThreadStatus.RUNNING, ThreadStatus.STOPPING)
            for job in self.job_list
        )

    def fetch_next_waiting(self) -> Optional[JobQueueNode]:
        for job in self.job_list:
            if job.thread_status == ThreadStatus.READY:
                return job
        return None

    def count_status(self, status: JobStatus) -> int:
        return len([job for job in self.job_list if job.queue_status == status])

    @property
    def stopped(self) -> bool:
        return self._stopped

    def kill_all_jobs(self) -> None:
        self._stopped = True

    @property
    def queue_size(self) -> int:
        return len(self.job_list)

    def add_job(self, job: JobQueueNode, iens: int) -> int:
        job.convertToCReference(None)
        queue_index: int = self._add_job(job)
        self.job_list.append(job)
        self._differ.add_state(queue_index, iens, job.queue_status.value)
        return queue_index

    def count_running(self) -> int:
        return sum(job.thread_status == ThreadStatus.RUNNING for job in self.job_list)

    def max_running(self) -> int:
        if self.get_max_running() == 0:
            return len(self.job_list)
        else:
            return self.get_max_running()

    def available_capacity(self) -> bool:
        return not self.stopped and self.count_running() < self.max_running()

    async def stop_jobs(self) -> None:
        for job in self.job_list:
            job.stop()
        while self.is_active():
            await asyncio.sleep(1)

    def assert_complete(self) -> None:
        for job in self.job_list:
            if job.thread_status != ThreadStatus.DONE:
                msg = (
                    "Unexpected job status type after "
                    "running job: {} with thread status: {}"
                )
                raise AssertionError(msg.format(job.queue_status, job.thread_status))

    def launch_jobs(self, pool_sema: Semaphore) -> None:
        # Start waiting jobs
        while self.available_capacity():
            job = self.fetch_next_waiting()
            if job is None:
                break
            job.run(
                driver=self.driver,
                pool_sema=pool_sema,
                max_submit=self.max_submit,
            )

    @staticmethod
    def _translate_change_to_cloudevent(
        ens_id: str, real_id: int, status: str
    ) -> CloudEvent:
        return CloudEvent(
            {
                "type": _queue_state_event_type(status),
                "source": f"/ert/ensemble/{ens_id}/real/{real_id}",
                "datacontenttype": "application/json",
            },
            {
                "queue_event_type": status,
            },
        )

    async def _publish_changes(
        self, changes: Dict[int, str], ee_connection: WebSocketClientProtocol
    ) -> None:
        assert self._ens_id is not None
        events = deque(
            [
                JobQueue._translate_change_to_cloudevent(self._ens_id, real_id, status)
                for real_id, status in changes.items()
            ]
        )
        while events:
            await ee_connection.send(to_json(events[0]))
            events.popleft()

    async def _jobqueue_publisher(self) -> None:
        assert self._changes_to_publish is not None
        ee_headers = Headers()
        if self._ee_token is not None:
            ee_headers["token"] = self._ee_token

        if self._ee_uri is None:
            # If no ensemble evaluator present, we will publish to the log
            while (
                change := await self._changes_to_publish.get()
            ) != CLOSE_PUBLISHER_SENTINEL:
                logger.warning(f"State change in jobqueue.execute(): {change}")
            return

        async for ee_connection in connect(
            self._ee_uri,
            ssl=self._ee_ssl_context,
            extra_headers=ee_headers,
            open_timeout=60,
            ping_timeout=60,
            ping_interval=60,
            close_timeout=60,
        ):
            try:
                while True:
                    change = await self._changes_to_publish.get()
                    if change == CLOSE_PUBLISHER_SENTINEL:
                        return
                    assert isinstance(change, dict)
                    await self._publish_changes(change, ee_connection)
            except ConnectionClosed:
                logger.debug(
                    "Websocket connection from JobQueue "
                    "to EnsembleEvaluator closed, will retry."
                )
                continue

    async def execute(
        self,
        min_required_realizations: int = 0,
    ) -> str:
        self._changes_to_publish = asyncio.Queue()
        asyncio.create_task(self._jobqueue_publisher())

        try:
            await self._changes_to_publish.put(self._differ.snapshot())
            while True:
                self.launch_jobs(self._pool_sema)

                await asyncio.sleep(1)

                if min_required_realizations > 0:
                    self.stop_long_running_jobs(min_required_realizations)

                changes, new_state = self.changes_without_transition()
                if len(changes) > 0:
                    await self._changes_to_publish.put(changes)
                    self._differ.transition_to_new_state(new_state)

                if self.stopped:
                    logger.debug("queue cancelled, stopping jobs...")
                    await self.stop_jobs()
                    await self._changes_to_publish.put(CLOSE_PUBLISHER_SENTINEL)
                    return EVTYPE_ENSEMBLE_CANCELLED

                if not self.is_active():
                    break

        except Exception:
            logger.exception(
                "unexpected exception in queue",
                exc_info=True,
            )
            await self.stop_jobs()
            logger.debug("jobs stopped, re-raising exception")
            return EVTYPE_ENSEMBLE_FAILED

        if not self.stopped:
            self.assert_complete()

            self._differ.transition(self.job_list)
            await self._changes_to_publish.put(self._differ.snapshot())
            await self._changes_to_publish.put(CLOSE_PUBLISHER_SENTINEL)

        return EVTYPE_ENSEMBLE_STOPPED

    # pylint: disable=too-many-arguments
    def add_job_from_run_arg(
        self,
        run_arg: "RunArg",
        job_script: str,
        max_runtime: Optional[int],
        num_cpu: int,
    ) -> None:
        job = JobQueueNode(
            job_script=job_script,
            num_cpu=num_cpu,
            run_arg=run_arg,
            max_runtime=max_runtime,
        )

        if job is None:
            return
        run_arg.queue_index = self.add_job(job, run_arg.iens)

    def add_realization(
        self,
        real: Realization,
    ) -> None:
        job = JobQueueNode(
            job_script=real.job_script,
            num_cpu=real.num_cpu,
            run_arg=real.run_arg,
            max_runtime=real.max_runtime,
            callback_timeout=self._on_timeout,
        )
        if job is None:
            raise ValueError("JobQueueNode constructor created None job")

        real.run_arg.queue_index = self.add_job(job, real.run_arg.iens)

    def stop_long_running_jobs(self, minimum_required_realizations: int) -> None:
        completed_jobs = [
            job for job in self.job_list if job.queue_status == JobStatus.DONE
        ]
        finished_realizations = len(completed_jobs)

        if not finished_realizations:
            job_nodes_status = ""
            for job in self.job_list:
                job_nodes_status += str(job)
            logger.error(
                f"Attempted to stop finished jobs when none was found in queue"
                f"{str(self)}, {job_nodes_status}"
            )
            return

        if finished_realizations < minimum_required_realizations:
            return

        average_runtime = (
            sum(job.runtime for job in completed_jobs) / finished_realizations
        )

        for job in self.job_list:
            if job.runtime > LONG_RUNNING_FACTOR * average_runtime:
                job.stop()

    def snapshot(self) -> Optional[Dict[int, str]]:
        """Return the whole state, or None if there was no snapshot."""
        return self._differ.snapshot()

    def changes_without_transition(self) -> Tuple[Dict[int, str], List[JobStatus]]:
        old_state, new_state = self._differ.get_old_and_new_state(self.job_list)
        return self._differ.diff_states(old_state, new_state), new_state

    def add_dispatch_information_to_jobs_file(
        self,
        experiment_id: Optional[str] = None,
    ) -> None:
        for q_index, q_node in enumerate(self.job_list):
            cert_path = f"{q_node.run_path}/{CERT_FILE}"
            if self._ee_cert is not None:
                with open(cert_path, "w", encoding="utf-8") as cert_file:
                    cert_file.write(str(self._ee_cert))
            with open(
                f"{q_node.run_path}/{JOBS_FILE}", "r+", encoding="utf-8"
            ) as jobs_file:
                data = json.load(jobs_file)

                data["ens_id"] = self._ens_id
                data["real_id"] = self._differ.qindex_to_iens(q_index)
                data["dispatch_url"] = self._ee_uri
                data["ee_token"] = self._ee_token
                data["ee_cert_path"] = cert_path if self._ee_cert is not None else None
                data["experiment_id"] = experiment_id

                jobs_file.seek(0)
                jobs_file.truncate()
                json.dump(data, jobs_file, indent=4)
