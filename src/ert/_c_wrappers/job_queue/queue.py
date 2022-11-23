"""
Module implementing a queue for managing external jobs.

"""

import asyncio
import json
import logging
import ssl
import threading
import time
from collections import deque
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Union

from cloudevents.conversion import to_json
from cloudevents.http import CloudEvent
from cwrap import BaseCClass
from websockets.client import connect
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosedError

import _ert_com_protocol
from ert._c_wrappers import ResPrototype
from ert._c_wrappers.job_queue.job_queue_node import JobQueueNode
from ert._c_wrappers.job_queue.job_status_type_enum import JobStatusType
from ert._c_wrappers.job_queue.queue_differ import QueueDiffer
from ert._c_wrappers.job_queue.thread_status_type_enum import ThreadStatus
from ert.constant_filenames import CERT_FILE, JOBS_FILE, ERROR_file, STATUS_file

if TYPE_CHECKING:
    from ert._c_wrappers.enkf.res_config import ResConfig
    from ert._c_wrappers.enkf.run_arg import RunArg


logger = logging.getLogger(__name__)

LONG_RUNNING_FACTOR = 1.25


_FM_STEP_FAILURE = "com.equinor.ert.forward_model_step.failure"
_FM_STEP_PENDING = "com.equinor.ert.forward_model_step.pending"
_FM_STEP_RUNNING = "com.equinor.ert.forward_model_step.running"
_FM_STEP_SUCCESS = "com.equinor.ert.forward_model_step.success"
_FM_STEP_UNKNOWN = "com.equinor.ert.forward_model_step.unknown"
_FM_STEP_WAITING = "com.equinor.ert.forward_model_step.waiting"


_queue_state_to_event_type_map = {
    "JOB_QUEUE_NOT_ACTIVE": _FM_STEP_WAITING,
    "JOB_QUEUE_WAITING": _FM_STEP_WAITING,
    "JOB_QUEUE_SUBMITTED": _FM_STEP_WAITING,
    "JOB_QUEUE_PENDING": _FM_STEP_PENDING,
    "JOB_QUEUE_RUNNING": _FM_STEP_RUNNING,
    "JOB_QUEUE_DONE": _FM_STEP_RUNNING,
    "JOB_QUEUE_EXIT": _FM_STEP_RUNNING,
    "JOB_QUEUE_IS_KILLED": _FM_STEP_FAILURE,
    "JOB_QUEUE_DO_KILL": _FM_STEP_FAILURE,
    "JOB_QUEUE_SUCCESS": _FM_STEP_SUCCESS,
    "JOB_QUEUE_RUNNING_DONE_CALLBACK": _FM_STEP_RUNNING,
    "JOB_QUEUE_RUNNING_EXIT_CALLBACK": _FM_STEP_RUNNING,
    "JOB_QUEUE_STATUS_FAILURE": _FM_STEP_UNKNOWN,
    "JOB_QUEUE_FAILED": _FM_STEP_FAILURE,
    "JOB_QUEUE_DO_KILL_NODE_FAILURE": _FM_STEP_FAILURE,
    "JOB_QUEUE_UNKNOWN": _FM_STEP_UNKNOWN,
}


def _queue_state_event_type(state: JobStatusType) -> str:
    return _queue_state_to_event_type_map[state]


class JobQueue(BaseCClass):
    # If the queue is created with size == 0 that means that it will
    # just grow as needed; for the queue layer to know when to exit
    # you must call the function submit_complete() when you have no
    # more jobs to submit.
    #
    # If the number of jobs is known in advance you can create the
    # queue with a finite value for size, in that case it is not
    # necessary to explitly inform the queue layer when all jobs have
    # been submitted.
    TYPE_NAME = "job_queue"
    _alloc = ResPrototype("void* job_queue_alloc( int , char* , char* )", bind=False)
    _free = ResPrototype("void job_queue_free( job_queue )")
    _set_max_job_duration = ResPrototype(
        "void job_queue_set_max_job_duration( job_queue , int)"
    )
    _set_driver = ResPrototype("void job_queue_set_driver( job_queue , void* )")
    _kill_job = ResPrototype("bool job_queue_kill_job( job_queue , int )")
    _iget_driver_data = ResPrototype(
        "void* job_queue_iget_driver_data( job_queue , int)"
    )

    _num_running = ResPrototype("int  job_queue_get_num_running( job_queue )")
    _num_complete = ResPrototype("int  job_queue_get_num_complete( job_queue )")
    _num_waiting = ResPrototype("int  job_queue_get_num_waiting( job_queue )")
    _num_pending = ResPrototype("int  job_queue_get_num_pending( job_queue )")

    _is_running = ResPrototype("bool job_queue_is_running( job_queue )")
    _submit_complete = ResPrototype("void job_queue_submit_complete( job_queue )")
    _get_max_submit = ResPrototype("int job_queue_get_max_submit(job_queue)")

    _get_exit_file = ResPrototype("char* job_queue_get_exit_file(job_queue)")
    _get_status_file = ResPrototype("char* job_queue_get_status_file(job_queue)")
    _add_job = ResPrototype("int job_queue_add_job_node(job_queue, job_queue_node)")

    def __repr__(self):
        nrun, ncom, nwait, npend = (
            self._num_running(),
            self._num_complete(),
            self._num_waiting(),
            self._num_pending(),
        )
        isrun = "running" if self.isRunning else "not running"
        cnt = (
            "%s, num_running=%d, num_complete=%d, "
            "num_waiting=%d, num_pending=%d, active=%d"
        )
        return self._create_repr(cnt % (isrun, nrun, ncom, nwait, npend, len(self)))

    def __init__(self, driver, max_submit=2, size=0):
        """
        Short doc...
        The @max_submit argument says how many times the job be submitted
        (including a failure)
              max_submit = 2: means that we can submit job once more
        The @size argument is used to say how many jobs the queue will
        run, in total.
              size = 0: That means that you do not tell the queue in
                advance how many jobs you have. The queue will just run
                all the jobs you add, but you have to inform the queue in
                some way that all jobs have been submitted. To achieve
                this you should call the submit_complete() method when all
                jobs have been submitted.#

              size > 0: The queue will know exactly how many jobs to run,
                and will continue until this number of jobs have completed
                - it is not necessary to call the submit_complete() method
                in this case.
        """

        self.job_list = []
        self._stopped = False
        c_ptr = self._alloc(max_submit, STATUS_file, ERROR_file)
        super().__init__(c_ptr)
        self.size = size

        self.driver = driver
        self._set_driver(driver.from_param(driver))
        self._differ = QueueDiffer()

    def kill_job(self, queue_index):
        """
        Will kill job nr @index.
        """
        self._kill_job(queue_index)

    def submit_complete(self):
        """
        Method to inform the queue that all jobs have been submitted.

        If the queue has been created with size == 0 the queue has no
        way of knowing when all jobs have completed; hence in that
        case you must call the submit_complete() method when all jobs
        have been submitted.

        If you know in advance exactly how many jobs you will run that
        should be specified with the size argument when creating the
        queue, in that case it is not necessary to call the
        submit_complete() method.
        """
        self._submit_complete()

    @property
    def isRunning(self):
        return self._is_running()

    @property
    def num_running(self):
        return self._num_running()

    @property
    def num_pending(self):
        return self._num_pending()

    @property
    def num_waiting(self):
        return self._num_waiting()

    @property
    def num_complete(self):
        return self._num_complete()

    def __getitem__(self, index):
        idx = index
        ls = len(self)
        if idx < 0:
            idx += ls
        if 0 <= idx < ls:
            return self._iget_driver_data(idx)
        raise IndexError(f"index out of range, was: {index} should be in [0, {ls})")

    def exists(self, index):
        return self[index]

    def get_max_running(self):
        return self.driver.get_max_running()

    def set_max_running(self, max_running):
        self.driver.set_max_running(max_running)

    def set_max_job_duration(self, max_duration: int) -> None:
        self._set_max_job_duration(max_duration)

    @property
    def max_submit(self):
        return self._get_max_submit()

    def free(self):
        self._free()

    def is_active(self):
        for job in self.job_list:
            if job.thread_status in (
                ThreadStatus.READY,
                ThreadStatus.RUNNING,
                ThreadStatus.STOPPING,
            ):
                return True
        return False

    def fetch_next_waiting(self):
        for job in self.job_list:
            if job.thread_status == ThreadStatus.READY:
                return job
        return None

    def count_status(self, status):
        return len([job for job in self.job_list if job.status == status])

    @property
    def stopped(self):
        return self._stopped

    def kill_all_jobs(self) -> None:
        self._stopped = True

    @property
    def queue_size(self):
        return len(self.job_list)

    @property
    def exit_file(self):
        return self._get_exit_file()

    @property
    def status_file(self):
        return self._get_status_file()

    def add_job(self, job, iens):
        job.convertToCReference(None)
        queue_index = self._add_job(job)
        self.job_list.append(job)
        self._differ.add_state(queue_index, iens, job.status.value)
        return queue_index

    def count_running(self):
        return sum(job.thread_status == ThreadStatus.RUNNING for job in self.job_list)

    def max_running(self):
        if self.get_max_running() == 0:
            return len(self.job_list)
        else:
            return self.get_max_running()

    def available_capacity(self):
        return not self.stopped and self.count_running() < self.max_running()

    def stop_jobs(self):
        for job in self.job_list:
            job.stop()
        while self.is_active():
            time.sleep(1)

    async def stop_jobs_async(self):
        for job in self.job_list:
            job.stop()
        while self.is_active():
            await asyncio.sleep(1)

    def assert_complete(self):
        for job in self.job_list:
            if job.thread_status != ThreadStatus.DONE:
                msg = (
                    "Unexpected job status type after "
                    "running job: {} with thread status: {}"
                )
                raise AssertionError(msg.format(job.status, job.thread_status))

    def launch_jobs(self, pool_sema):
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

    def execute_queue(self, pool_sema, evaluators):
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
        ens_id: str, real_id: str, status: JobStatusType
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
        experiment_id: str, ens_id: str, real_id: str, status: JobStatusType
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
        changes,
        output_bus: "asyncio.Queue[_ert_com_protocol.DispatcherMessage]",
    ):
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
        changes,
        ws_uri: str,
        ssl_context: ssl.SSLContext,
        headers: Mapping[str, str],
    ):
        events = deque(
            [
                JobQueue._translate_change_to_cloudevent(ens_id, real_id, status)
                for real_id, status in changes.items()
            ]
        )

        retries = 0
        while True:
            try:
                async with connect(
                    ws_uri, ssl=ssl_context, extra_headers=headers
                ) as websocket:
                    while events:
                        await asyncio.wait_for(websocket.send(to_json(events[0])), 60)
                        events.popleft()
                    return
            except (ConnectionClosedError, asyncio.TimeoutError) as e:
                if retries >= 10:
                    logger.exception(
                        "Connection to websocket %s failed, unable to publish changes",
                        ws_uri,
                    )
                    raise

                # websockets for python > 3.6 comes with builtin backoff, implement a
                # crude one here
                retries += 1
                backoff = max(3, min(60, 2**retries))
                logger.info(
                    "Connection to websocket %s was closed, retry in %d seconds.",
                    ws_uri,
                    backoff,
                    exc_info=e,
                )

                await asyncio.sleep(backoff)

    async def execute_queue_via_websockets(  # pylint: disable=too-many-arguments
        self,
        ws_uri: str,
        ens_id: str,
        pool_sema: threading.BoundedSemaphore,
        evaluators: List[Callable[..., Any]],
        cert: Optional[Union[str, bytes]] = None,
        token: Optional[str] = None,
    ) -> None:
        if evaluators is None:
            evaluators = []
        if cert is not None:
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            ssl_context.load_verify_locations(cadata=cert)
        else:
            ssl_context = True if ws_uri.startswith("wss") else None
        headers = Headers()
        if token is not None:
            headers["token"] = token

        try:
            await JobQueue._publish_changes(
                ens_id, self._differ.snapshot(), ws_uri, ssl_context, headers
            )
            while True:
                self.launch_jobs(pool_sema)

                await asyncio.sleep(1)

                for func in evaluators:
                    func()

                await JobQueue._publish_changes(
                    ens_id,
                    self.changes_after_transition(),
                    ws_uri,
                    ssl_context,
                    headers,
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
        await JobQueue._publish_changes(
            ens_id, self._differ.snapshot(), ws_uri, ssl_context, headers
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
        res_config: "ResConfig",
        max_runtime: Optional[int],
        ok_cb: Callable[..., Any],
        exit_cb: Callable[..., Any],
        num_cpu: int,
    ) -> None:
        job_name = run_arg.job_name
        run_path = run_arg.runpath
        job_script = res_config.queue_config.job_script

        job = JobQueueNode(
            job_script=job_script,
            job_name=job_name,
            run_path=run_path,
            num_cpu=num_cpu,
            status_file=self.status_file,
            exit_file=self.exit_file,
            done_callback_function=ok_cb,
            exit_callback_function=exit_cb,
            callback_arguments=[run_arg, res_config],
            max_runtime=max_runtime,
        )

        if job is None:
            return
        run_arg.set_queue_index(self.add_job(job, run_arg.iens))

    def add_ee_stage(self, stage, callback_timeout=None):
        job = JobQueueNode(
            job_script=stage.job_script,
            job_name=stage.job_name,
            run_path=str(stage.run_path),
            num_cpu=stage.num_cpu,
            status_file=self.status_file,
            exit_file=self.exit_file,
            done_callback_function=stage.done_callback,
            exit_callback_function=stage.exit_callback,
            callback_arguments=stage.callback_arguments,
            max_runtime=stage.max_runtime,
            callback_timeout=callback_timeout,
        )
        if job is None:
            raise ValueError("JobQueueNode constructor created None job")

        iens = stage.run_arg.iens
        stage.run_arg.set_queue_index(self.add_job(job, iens))

    def stop_long_running_jobs(self, minimum_required_realizations: int) -> None:
        finished_realizations = self.count_status(JobStatusType.JOB_QUEUE_DONE)
        if finished_realizations < minimum_required_realizations:
            return

        completed_jobs = [
            job for job in self.job_list if job.status == JobStatusType.JOB_QUEUE_DONE
        ]
        average_runtime = sum(job.runtime for job in completed_jobs) / len(
            completed_jobs
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

    def add_dispatch_information_to_jobs_file(
        self,
        ens_id: str,
        dispatch_url: str,
        cert: Optional[Union[str, bytes]],
        token: Optional[str],
        experiment_id: Optional[str] = None,
    ) -> None:
        for q_index, q_node in enumerate(self.job_list):
            if cert is not None:
                cert_path = f"{q_node.run_path}/{CERT_FILE}"
                with open(cert_path, "w") as cert_file:
                    cert_file.write(cert)
            with open(f"{q_node.run_path}/{JOBS_FILE}", "r+") as jobs_file:
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
