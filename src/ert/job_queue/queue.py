"""
Module implementing a queue for managing external jobs.

"""
from __future__ import annotations

import asyncio
import json
import logging
import ssl
import threading
import time
from collections import deque
from threading import Semaphore
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

from cloudevents.conversion import to_json
from cloudevents.http import CloudEvent
from cwrap import BaseCClass
from websockets.client import WebSocketClientProtocol, connect
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed

import _ert_com_protocol
from ert.constant_filenames import CERT_FILE, JOBS_FILE, ERROR_file, STATUS_file
from ert.job_queue.job_queue_node import JobQueueNode
from ert.job_queue.job_status import JobStatus
from ert.job_queue.queue_differ import QueueDiffer
from ert.job_queue.thread_status import ThreadStatus

from . import ResPrototype

if TYPE_CHECKING:
    from ert.config import ErtConfig
    from ert.ensemble_evaluator import LegacyStep
    from ert.run_arg import RunArg

    from .driver import Driver


logger = logging.getLogger(__name__)

LONG_RUNNING_FACTOR = 1.25


_FM_STEP_FAILURE = "com.equinor.ert.forward_model_step.failure"
_FM_STEP_PENDING = "com.equinor.ert.forward_model_step.pending"
_FM_STEP_RUNNING = "com.equinor.ert.forward_model_step.running"
_FM_STEP_SUCCESS = "com.equinor.ert.forward_model_step.success"
_FM_STEP_UNKNOWN = "com.equinor.ert.forward_model_step.unknown"
_FM_STEP_WAITING = "com.equinor.ert.forward_model_step.waiting"


_queue_state_to_event_type_map = {
    "NOT_ACTIVE": _FM_STEP_WAITING,
    "WAITING": _FM_STEP_WAITING,
    "SUBMITTED": _FM_STEP_WAITING,
    "PENDING": _FM_STEP_PENDING,
    "RUNNING": _FM_STEP_RUNNING,
    "DONE": _FM_STEP_RUNNING,
    "EXIT": _FM_STEP_RUNNING,
    "IS_KILLED": _FM_STEP_FAILURE,
    "DO_KILL": _FM_STEP_FAILURE,
    "SUCCESS": _FM_STEP_SUCCESS,
    "RUNNING_DONE_CALLBACK": _FM_STEP_RUNNING,
    "RUNNING_EXIT_CALLBACK": _FM_STEP_RUNNING,
    "STATUS_FAILURE": _FM_STEP_UNKNOWN,
    "FAILED": _FM_STEP_FAILURE,
    "DO_KILL_NODE_FAILURE": _FM_STEP_FAILURE,
    "UNKNOWN": _FM_STEP_UNKNOWN,
}


def _queue_state_event_type(state: str) -> str:
    return _queue_state_to_event_type_map[state]


# pylint: disable=too-many-public-methods
class JobQueue(BaseCClass):  # type: ignore
    TYPE_NAME = "job_queue"
    _alloc = ResPrototype("void* job_queue_alloc()", bind=False)
    _free = ResPrototype("void job_queue_free( job_queue )")
    _set_driver = ResPrototype("void job_queue_set_driver( job_queue , void* )")

    _add_job = ResPrototype("int job_queue_add_job_node(job_queue, job_queue_node)")

    def __repr__(self) -> str:
        return f"JobQueue({self.driver}, {self.max_submit})"

    def __str__(self) -> str:
        return self.__repr__()

    def __init__(self, driver: "Driver", max_submit: int = 2):
        """
        Short doc...
        The @max_submit argument says how many times the job should be submitted
        (including a failure)
              max_submit = 2: means that we can submit job once more
        The @size argument is used to say how many jobs the queue will
        run, in total.
        """

        self.job_list: List[JobQueueNode] = []
        self._stopped = False
        c_ptr = self._alloc(STATUS_file, ERROR_file)
        super().__init__(c_ptr)

        self.driver = driver
        self._set_driver(driver.from_param(driver))
        self._differ = QueueDiffer()
        self._max_job_duration = 0
        self._max_submit = max_submit

    def get_max_running(self) -> int:
        return self.driver.get_max_running()

    def set_max_running(self, max_running: int) -> None:
        self.driver.set_max_running(max_running)

    def set_max_job_duration(self, max_duration: int) -> None:
        self._max_job_duration = max_duration

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
        return len([job for job in self.job_list if job.status == status])

    @property
    def stopped(self) -> bool:
        return self._stopped

    def kill_all_jobs(self) -> None:
        self._stopped = True

    @property
    def queue_size(self) -> int:
        return len(self.job_list)

    @property
    def exit_file(self) -> str:
        return ERROR_file

    @property
    def status_file(self) -> str:
        return STATUS_file

    def add_job(self, job: JobQueueNode, iens: int) -> int:
        job.convertToCReference(None)
        queue_index: int = self._add_job(job)
        self.job_list.append(job)
        self._differ.add_state(queue_index, iens, job.status.value)
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

    def stop_jobs(self) -> None:
        for job in self.job_list:
            job.stop()
        while self.is_active():
            time.sleep(1)

    async def stop_jobs_async(self) -> None:
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
                raise AssertionError(msg.format(job.status, job.thread_status))

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

    def execute_queue(
        self, pool_sema: Semaphore, evaluators: Optional[Iterable[Callable[[], None]]]
    ) -> None:
        while self.is_active() and not self.stopped:
            self.launch_jobs(pool_sema)

            time.sleep(1)

            if evaluators is not None:
                for func in evaluators:
                    func()

        if self.stopped:
            self.stop_jobs()

        self.assert_complete()

    @staticmethod
    def _translate_change_to_cloudevent(
        ens_id: str, real_id: int, status: str
    ) -> CloudEvent:
        return CloudEvent(
            {
                "type": _queue_state_event_type(status),
                "source": f"/ert/ensemble/{ens_id}/real/{real_id}/step/{0}",
                "datacontenttype": "application/json",
            },
            {
                "queue_event_type": status,
            },
        )

    @staticmethod
    def _translate_change_to_protobuf(
        experiment_id: str, ens_id: str, real_id: int, status: str
    ) -> _ert_com_protocol.DispatcherMessage:
        return _ert_com_protocol.node_status_builder(
            status=_ert_com_protocol.queue_state_to_pbuf_type(status),
            experiment_id=experiment_id,
            ensemble_id=ens_id,
            realization_id=real_id,
            step_id=0,
        )

    @staticmethod
    async def _queue_changes(
        experiment_id: str,
        ens_id: str,
        changes: Dict[int, str],
        output_bus: "asyncio.Queue[_ert_com_protocol.DispatcherMessage]",
    ) -> None:
        events = [
            JobQueue._translate_change_to_protobuf(
                experiment_id, ens_id, real_id, status
            )
            for real_id, status in changes.items()
        ]

        for event in events:
            output_bus.put_nowait(event)

    @staticmethod
    async def _publish_changes(
        ens_id: str,
        changes: Dict[int, str],
        ee_connection: WebSocketClientProtocol,
    ) -> None:
        events = deque(
            [
                JobQueue._translate_change_to_cloudevent(ens_id, real_id, status)
                for real_id, status in changes.items()
            ]
        )
        while events:
            await ee_connection.send(to_json(events[0]))
            events.popleft()

    async def _execution_loop_queue_via_websockets(
        self,
        ee_connection: WebSocketClientProtocol,
        ens_id: str,
        pool_sema: threading.BoundedSemaphore,
        evaluators: List[Callable[..., Any]],
    ) -> None:
        while True:
            self.launch_jobs(pool_sema)

            await asyncio.sleep(1)

            for func in evaluators:
                func()

            changes, new_state = self.changes_without_transition()
            # logically not necessary the way publish changes is implemented at the
            # moment, but highly relevant before, and might be relevant in the
            # future in case publish changes becomes expensive again
            if len(changes) > 0:
                await JobQueue._publish_changes(
                    ens_id,
                    changes,
                    ee_connection,
                )
                self._differ.transition_to_new_state(new_state)

            if self.stopped:
                raise asyncio.CancelledError

            if not self.is_active():
                break

    async def execute_queue_via_websockets(  # pylint: disable=too-many-arguments
        self,
        ee_uri: str,
        ens_id: str,
        pool_sema: threading.BoundedSemaphore,
        evaluators: List[Callable[..., Any]],
        ee_cert: Optional[Union[str, bytes]] = None,
        ee_token: Optional[str] = None,
    ) -> None:
        if evaluators is None:
            evaluators = []
        ee_ssl_context: Optional[Union[ssl.SSLContext, bool]]
        if ee_cert is not None:
            ee_ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            ee_ssl_context.load_verify_locations(cadata=ee_cert)
        else:
            ee_ssl_context = True if ee_uri.startswith("wss") else None
        ee_headers = Headers()
        if ee_token is not None:
            ee_headers["token"] = ee_token

        try:
            # initial publish
            async with connect(
                ee_uri,
                ssl=ee_ssl_context,
                extra_headers=ee_headers,
                open_timeout=60,
                ping_timeout=60,
                ping_interval=60,
                close_timeout=60,
            ) as ee_connection:
                await JobQueue._publish_changes(
                    ens_id, self._differ.snapshot(), ee_connection
                )
            # loop
            async for ee_connection in connect(
                ee_uri,
                ssl=ee_ssl_context,
                extra_headers=ee_headers,
                open_timeout=60,
                ping_timeout=60,
                ping_interval=60,
                close_timeout=60,
            ):
                try:
                    await self._execution_loop_queue_via_websockets(
                        ee_connection, ens_id, pool_sema, evaluators
                    )
                except ConnectionClosed:
                    logger.warning(
                        "job queue dropped connection to ensemble evaulator - "
                        "going to try and reconnect"
                    )
                    continue
                if not self.is_active():
                    break

        except asyncio.CancelledError:
            logger.debug("queue cancelled, stopping jobs...")
            await self.stop_jobs_async()
            logger.debug("jobs stopped, re-raising CancelledError")
            raise

        except Exception:
            logger.exception(
                "unexpected exception in queue",
                exc_info=True,
            )
            await self.stop_jobs_async()
            logger.debug("jobs stopped, re-raising exception")
            raise

        self.assert_complete()
        self._differ.transition(self.job_list)
        # final publish
        async with connect(
            ee_uri,
            ssl=ee_ssl_context,
            extra_headers=ee_headers,
            open_timeout=60,
            ping_timeout=60,
            ping_interval=60,
            close_timeout=60,
        ) as ee_connection:
            await JobQueue._publish_changes(
                ens_id, self._differ.snapshot(), ee_connection
            )

    async def execute_queue_comms_via_bus(  # pylint: disable=too-many-arguments
        self,
        experiment_id: str,
        ens_id: str,
        pool_sema: threading.BoundedSemaphore,
        evaluators: List[Callable[..., Any]],
        output_bus: "asyncio.Queue[_ert_com_protocol.DispatcherMessage]",
    ) -> None:
        if evaluators is None:
            evaluators = []
        try:
            await JobQueue._queue_changes(
                experiment_id, ens_id, self._differ.snapshot(), output_bus
            )
            while True:
                self.launch_jobs(pool_sema)

                await asyncio.sleep(1)

                for func in evaluators:
                    func()

                changes = self.changes_after_transition()
                await JobQueue._queue_changes(
                    experiment_id, ens_id, changes, output_bus
                )

                if self.stopped:
                    raise asyncio.CancelledError

                if not self.is_active():
                    break

        except asyncio.CancelledError:
            logger.debug("queue cancelled, stopping jobs...")
            await self.stop_jobs_async()
            logger.debug("jobs stopped, re-raising CancelledError")
            raise

        except Exception:
            logger.exception(
                "unexpected exception in queue",
                exc_info=True,
            )
            await self.stop_jobs_async()
            logger.debug("jobs stopped, re-raising exception")
            raise

        self.assert_complete()
        self._differ.transition(self.job_list)
        await JobQueue._queue_changes(
            experiment_id, ens_id, self._differ.snapshot(), output_bus
        )

    # pylint: disable=too-many-arguments
    def add_job_from_run_arg(
        self,
        run_arg: "RunArg",
        ert_config: "ErtConfig",
        max_runtime: Optional[int],
        num_cpu: int,
    ) -> None:
        job_name = run_arg.job_name
        run_path = run_arg.runpath
        job_script = ert_config.queue_config.job_script

        job = JobQueueNode(
            job_script=job_script,
            job_name=job_name,
            run_path=run_path,
            num_cpu=num_cpu,
            status_file=self.status_file,
            exit_file=self.exit_file,
            run_arg=run_arg,
            ensemble_config=ert_config.ensemble_config,
            max_runtime=max_runtime,
        )

        if job is None:
            return
        run_arg.queue_index = self.add_job(job, run_arg.iens)

    def add_ee_stage(
        self,
        stage: "LegacyStep",
        callback_timeout: Optional[Callable[[int], None]] = None,
    ) -> None:
        job = JobQueueNode(
            job_script=stage.job_script,
            job_name=stage.job_name,
            run_path=str(stage.run_path),
            num_cpu=stage.num_cpu,
            status_file=self.status_file,
            exit_file=self.exit_file,
            run_arg=stage.run_arg,
            ensemble_config=stage.ensemble_config,
            max_runtime=stage.max_runtime,
            callback_timeout=callback_timeout,
        )
        if job is None:
            raise ValueError("JobQueueNode constructor created None job")

        iens = stage.run_arg.iens
        stage.run_arg.queue_index = self.add_job(job, iens)

    def stop_long_running_jobs(self, minimum_required_realizations: int) -> None:
        completed_jobs = [job for job in self.job_list if job.status == JobStatus.DONE]
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

    def changes_after_transition(self) -> Dict[int, str]:
        old_state, new_state = self._differ.transition(self.job_list)
        return self._differ.diff_states(old_state, new_state)

    def changes_without_transition(self) -> Tuple[Dict[int, str], List[JobStatus]]:
        old_state, new_state = self._differ.get_old_and_new_state(self.job_list)
        return self._differ.diff_states(old_state, new_state), new_state

    def add_dispatch_information_to_jobs_file(
        self,
        ens_id: str,
        dispatch_url: str,
        cert: Optional[str],
        token: Optional[str],
        experiment_id: Optional[str] = None,
    ) -> None:
        for q_index, q_node in enumerate(self.job_list):
            cert_path = f"{q_node.run_path}/{CERT_FILE}"
            if cert is not None:
                with open(cert_path, "w", encoding="utf-8") as cert_file:
                    cert_file.write(cert)
            with open(
                f"{q_node.run_path}/{JOBS_FILE}", "r+", encoding="utf-8"
            ) as jobs_file:
                data = json.load(jobs_file)

                data["ens_id"] = ens_id
                data["real_id"] = self._differ.qindex_to_iens(q_index)
                data["step_id"] = 0
                data["dispatch_url"] = dispatch_url
                data["ee_token"] = token
                data["ee_cert_path"] = cert_path if cert is not None else None
                data["experiment_id"] = experiment_id

                jobs_file.seek(0)
                jobs_file.truncate()
                json.dump(data, jobs_file, indent=4)
