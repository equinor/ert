"""
Module implementing a queue for managing external jobs.

"""
from __future__ import annotations

import asyncio
import datetime
import json
import logging
import ssl
from collections import deque
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

from cloudevents.conversion import to_json
from cloudevents.http import CloudEvent
from websockets.client import WebSocketClientProtocol, connect
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed

from ert.config import QueueConfig
from ert.constant_filenames import CERT_FILE, JOBS_FILE

from .driver import Driver
from .realization_state import QueueableRealization, RealizationState

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

EVTYPE_REALIZATION_FAILURE = "com.equinor.ert.realization.failure"
EVTYPE_REALIZATION_PENDING = "com.equinor.ert.realization.pending"
EVTYPE_REALIZATION_RUNNING = "com.equinor.ert.realization.running"
EVTYPE_REALIZATION_SUCCESS = "com.equinor.ert.realization.success"
EVTYPE_REALIZATION_UNKNOWN = "com.equinor.ert.realization.unknown"
EVTYPE_REALIZATION_WAITING = "com.equinor.ert.realization.waiting"
EVTYPE_REALIZATION_TIMEOUT = "com.equinor.ert.realization.timeout"
EVTYPE_ENSEMBLE_STARTED = "com.equinor.ert.ensemble.started"
EVTYPE_ENSEMBLE_STOPPED = "com.equinor.ert.ensemble.stopped"
EVTYPE_ENSEMBLE_CANCELLED = "com.equinor.ert.ensemble.cancelled"
EVTYPE_ENSEMBLE_FAILED = "com.equinor.ert.ensemble.failed"

_queue_state_to_event_type_map = {
    # NB, "active" is misleading, because realizations not selected (aka deactivated in the GUI) by the user will not be supplied here.
    "NOT_ACTIVE": EVTYPE_REALIZATION_WAITING,
    "WAITING": EVTYPE_REALIZATION_WAITING,
    "SUBMITTED": EVTYPE_REALIZATION_WAITING,  # a microstate not visible in the monitor
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


class JobQueue:
    """Represents a queue of realizations (aka Jobs) to be executed on a
    cluster."""

    def __init__(self, queue_config: QueueConfig):
        self._realizations: List[RealizationState] = []
        self.driver: Driver = Driver.create_driver(queue_config)
        self._queue_config = queue_config

        self._queue_stopped = False

        # Wrap these in a dataclass?
        self._ens_id: Optional[str] = None
        self._ee_uri: Optional[str] = None
        self._ee_cert: Optional[Union[str, bytes]] = None
        self._ee_token: Optional[str] = None
        self._ee_ssl_context: Optional[Union[ssl.SSLContext, bool]] = None

        self._changes_to_publish: Optional[
            asyncio.Queue[Union[Dict[int, str], object]]
        ] = None

    def is_active(self) -> bool:
        return any(
            real.current_state
            in (
                RealizationState.WAITING,
                RealizationState.SUBMITTED,
                RealizationState.PENDING,
                RealizationState.RUNNING,
                RealizationState.DONE,
            )
            for real in self._realizations
        )

    def count_status(self, state: RealizationState) -> int:
        return len([real for real in self._realizations if real.current_state == state])

    async def run_done_callback(self, state: RealizationState):
        state.validate()

    @property
    def stopped(self) -> bool:
        return self._queue_stopped

    async def stop_jobs_async(self) -> None:
        self.kill_all_jobs()
        # Wait until all kill commands are acknowlegded by the driver
        while any(
            (
                real
                for real in self._realizations
                if real.current_state
                not in (
                    RealizationState.IS_KILLED,
                    RealizationState.DO_KILL_NODE_FAILURE,
                )
            )
        ):
            await asyncio.sleep(0.1)

    def kill_all_jobs(self) -> None:
        for real in self._realizations:
            if real.current_state not in (
                RealizationState.DO_KILL,
                RealizationState.IS_KILLED,
            ):
                real.dokill()  # Initiates async killing

    @property
    def queue_size(self) -> int:
        return len(self._realizations)

    def _add_realization(self, realization: QueueableRealization) -> None:
        self._realizations.append(RealizationState(self, realization, retries=1))

    def count_running(self) -> int:
        return sum(
            real.current_state == RealizationState.RUNNING
            for real in self._realizations
        )

    def max_running(self) -> int:
        max_running = 0
        if self.driver.has_option("MAX_RUNNING"):
            max_running = int(self.driver.get_option("MAX_RUNNING"))
        if max_running == 0:
            return len(self._realizations)
        return max_running

    def available_capacity(self) -> bool:
        return self.count_running() < self.max_running()

    def all_success(self) -> bool:
        return all(
            real.current_state == RealizationState.SUCCESS for real in self._realizations)

    async def launch_jobs(self) -> None:
        while self.available_capacity():
            try:
                realization = next(
                    (
                        real
                        for real in self._realizations
                        if real.current_state == RealizationState.WAITING
                    )
                )
                realization.submit()
            except StopIteration:
                break

    def set_ee_info(
        self,
        ee_uri: str,
        ens_id: str,
        ee_cert: Optional[Union[str, bytes]] = None,
        ee_token: Optional[str] = None,
        verify_context: bool = True,
    ) -> None:
        self._ens_id = ens_id
        self._ee_token = ee_token

        self._ee_uri = ee_uri
        if ee_cert is not None:
            self._ee_cert = ee_cert
            self._ee_token = ee_token
            self._ee_ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            if verify_context:
                self._ee_ssl_context.load_verify_locations(cadata=ee_cert)
        else:
            self._ee_ssl_context = True if ee_uri.startswith("wss") else None

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
                        print("CLOSE SENTINEL")
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
        evaluators: List[Callable[..., Any]],
    ) -> None:
        if evaluators is None:
            evaluators = []

        self._changes_to_publish = asyncio.Queue()
        asyncio.create_task(self._jobqueue_publisher())

        try:
            # await self._changes_to_publish.put(self._differ.snapshot())  # Reimplement me!, maybe send waiting states?
            while True:
                await self.launch_jobs()

                await asyncio.sleep(2)

                for func in evaluators:
                    func()

                await self.driver.poll_statuses()

                for real in self._realizations:
                    if (
                        real.current_state == RealizationState.RUNNING
                        and real.start_time
                        and datetime.datetime.now() - real.start_time
                        > datetime.timedelta(seconds=real.realization.max_runtime)
                    ):
                        real.dokill()

                if self.stopped:
                    print("WE ARE STOPPED")
                    logger.debug("queue cancelled, stopping jobs...")
                    await self.stop_jobs_async()
                    await self._changes_to_publish.put(CLOSE_PUBLISHER_SENTINEL)
                    return EVTYPE_ENSEMBLE_CANCELLED

                if not self.is_active():
                    print("not active, breaking out")
                    await asyncio.sleep(0.1)  # Let changes be propagated to the queue
                    while not self._changes_to_publish.empty():
                        # Drain queue..
                        await asyncio.sleep(0.1)
                    break

        except Exception as exc:
            print("EXCEPTION HAPPENED")
            print(exc)
            logger.exception(
                "unexpected exception in queue",
                exc_info=True,
            )
            await self.stop_jobs_async()
            logger.debug("jobs stopped, re-raising exception")
            return EVTYPE_ENSEMBLE_FAILED

        await self._changes_to_publish.put(CLOSE_PUBLISHER_SENTINEL)

        if not self.all_success():
            return EVTYPE_ENSEMBLE_FAILED

        return EVTYPE_ENSEMBLE_STOPPED

    def add_realization_from_run_arg(
        self,
        run_arg: "RunArg",
        job_script: str,
        max_runtime: Optional[int],
        num_cpu: int,
    ) -> None:
        qreal = QueueableRealization(
            iens=run_arg.iens,
            job_script=job_script,
            run_arg=run_arg,
            num_cpu=num_cpu,
            max_runtime=max_runtime,
        )
        # Everest uses this queue_index?
        run_arg.queue_index = self._add_realization(qreal)

    def add_realization(
        self,
        real: Realization,  # ensemble_evaluator.Realization
        callback_timeout: Optional[Callable[[int], None]] = None,
    ) -> None:
        qreal = QueueableRealization(
            job_script=real.job_script,
            num_cpu=real.num_cpu,
            run_arg=real.run_arg,
            max_runtime=real.max_runtime,
            callback_timeout=callback_timeout,
        )
        # Everest uses this queue_index?
        real.run_arg.queue_index = self._add_realization(qreal)

    def stop_long_running_realizations(
        self, minimum_required_realizations: int
    ) -> None:
        completed = [
            real
            for real in self._realizations
            if real.current_state == RealizationState.DONE
        ]
        finished_realizations = len(completed)

        if not finished_realizations:
            real_states = [str(real.current_state) for real in self._realizations].join(
                ","
            )
            logger.error(
                f"Attempted to stop finished realizations before any realization is finished"
                f"{real_states}"
            )
            return

        if finished_realizations < minimum_required_realizations:
            return

        average_runtime = (
            sum(real.runtime for real in completed) / finished_realizations
        )

        for job in self.job_list:
            if job.runtime > LONG_RUNNING_FACTOR * average_runtime:
                job.stop()

    def snapshot(self) -> Optional[Dict[int, str]]:
        """Return the whole state, or None if there was no snapshot."""
        return self._differ.snapshot()

    def add_dispatch_information_to_jobs_file(
        self,
        experiment_id: Optional[str] = None,
    ) -> None:
        for job in self._realizations:
            cert_path = f"{job.realization.run_arg.runpath}/{CERT_FILE}"
            if self._ee_cert is not None:
                with open(cert_path, "w", encoding="utf-8") as cert_file:
                    cert_file.write(str(self._ee_cert))
            with open(
                f"{job.realization.run_arg.runpath}/{JOBS_FILE}", "r+", encoding="utf-8"
            ) as jobs_file:
                data = json.load(jobs_file)

                data["ens_id"] = self._ens_id
                data["real_id"] = job.realization.run_arg.iens
                data["dispatch_url"] = self._ee_uri
                data["ee_token"] = self._ee_token
                data["ee_cert_path"] = cert_path if self._ee_cert is not None else None
                data["experiment_id"] = experiment_id

                jobs_file.seek(0)
                jobs_file.truncate()
                json.dump(data, jobs_file, indent=4)
