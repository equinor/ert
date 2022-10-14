import asyncio
import contextlib
import functools
import importlib
import logging
import multiprocessing
import os
import signal
import threading
import time
import warnings
from datetime import timedelta
from multiprocessing.process import BaseProcess
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Union

import cloudpickle
from cloudevents.conversion import to_json
from cloudevents.http import CloudEvent

with warnings.catch_warnings():
    # This is needed until dask_jobqueue is tagged by something newer
    # than 0.7.3
    warnings.filterwarnings("ignore", category=FutureWarning)
    from dask_jobqueue.lsf import LSFJob

import prefect
import prefect.utilities.logging
from prefect import Flow  # type: ignore
from prefect import context as prefect_context  # type: ignore
from prefect.engine.state import State
from prefect.executors import DaskExecutor, LocalDaskExecutor  # type: ignore

from _ert_job_runner.client import Client
from ert.async_utils import get_event_loop
from ert.ensemble_evaluator import identifiers
from ert.ensemble_evaluator.config import EvaluatorConnectionInfo
from ert.shared.port_handler import find_available_port

from ._ensemble import Ensemble
from ._function_task import FunctionTask
from ._io_map import _ensemble_transmitter_mapping
from ._realization import Realization
from ._step import FunctionStep, UnixStep
from ._unix_task import UnixTask

if TYPE_CHECKING:
    from multiprocessing.context import ForkServerContext

    from ert.ensemble_evaluator import EvaluatorServerConfig

DEFAULT_MAX_RETRIES = 0
DEFAULT_RETRY_DELAY = 5  # seconds


logger = logging.getLogger(__name__)


@contextlib.contextmanager
def prefect_log_level_context(level: Union[int, str]) -> Iterator[None]:
    prefect_logger = prefect.utilities.logging.get_logger()
    prev_log_level = prefect_logger.level
    prefect_logger.setLevel(level=level)
    yield
    prefect_logger.setLevel(level=prev_log_level)


async def _eq_submit_job(self: Any, script_filename: str) -> Any:
    with open(script_filename, encoding="utf-8") as fh:
        lines = fh.readlines()[1:]
    lines = [
        line.strip() if "#BSUB" not in line else line[5:].strip() for line in lines
    ]
    piped_cmd = [self.submit_command + " ".join(lines)]
    return self._call(piped_cmd, shell=True)


def _get_executor(custom_port_range: Optional[range], name: str = "local") -> Any:
    # See https://github.com/equinor/ert/pull/2757#discussion_r794368854
    _, port, sock = find_available_port(custom_range=custom_port_range)
    sock.close()  # do this explicitly, not relying on GC

    cluster_kwargs: Dict[str, Any] = {}

    if name == "local":
        cluster_kwargs = {
            "silence_logs": "debug",
        }
        return LocalDaskExecutor(**cluster_kwargs)
    elif name == "lsf":
        LSFJob._submit_job = _eq_submit_job
        cluster_kwargs = {
            "queue": "mr",
            "project": None,
            "cores": 1,
            "memory": "1GB",
            "use_stdin": True,
            "n_workers": 2,
            "silence_logs": "debug",
            "scheduler_options": {"port": port},
        }
        return DaskExecutor(
            cluster_class="dask_jobqueue.LSFCluster",
            cluster_kwargs=cluster_kwargs,
            debug=True,
        )
    elif name == "pbs":
        cluster_kwargs = {
            "n_workers": 10,
            "queue": "normal",
            "project": "ERT-TEST",
            "local_directory": "$TMPDIR",
            "cores": 1,
            "memory": "32gb",
            "resource_spec": "select=1:ncpus=1:mem=32gb",
            "scheduler_options": {"port": port},
            "extra": ["--worker-port", "51820:51840"],
        }
        return DaskExecutor(
            cluster_class="dask_jobqueue.PBSCluster",
            cluster_kwargs=cluster_kwargs,
            debug=True,
        )
    else:
        raise ValueError(f"Unknown executor name {name}")


# pylint: disable=no-member
def _on_task_failure(
    task: Union[UnixTask, FunctionTask], state: State, ens_id: str
) -> None:
    if prefect_context.task_run_count > task.max_retries:
        url = prefect_context.url
        token = prefect_context.token
        cert = prefect_context.cert
        with Client(url, token, cert) as c:
            event = CloudEvent(
                {
                    "type": identifiers.EVTYPE_FM_STEP_FAILURE,
                    "source": task.step.source(),
                    "datacontenttype": "application/json",
                },
                {"error_msg": state.message},
            )
            c.send(to_json(event).decode())


class PrefectEnsemble(Ensemble):  # pylint: disable=too-many-instance-attributes
    def __init__(  # pylint: disable=too-many-arguments
        self,
        reals: List[Realization],
        inputs: _ensemble_transmitter_mapping,
        outputs: _ensemble_transmitter_mapping,
        max_running: int,
        max_retries: int,
        executor: str,
        retry_delay: int,
        id_: str,
        custom_port_range: Optional[range] = None,
    ):
        super().__init__(reals=reals, metadata={"iter": 0}, id_=id_)
        self._inputs = inputs
        self._outputs = outputs
        self._real_per_batch = max_running

        self._max_retries = max_retries
        self._retry_delay = timedelta(seconds=retry_delay)

        # If we instantiate an executor here the prefect ensemble
        # will fail to pickle (required when using multiprocessing),
        # as the executor has an internal thread lock. Hence, we
        # bind the parameters and delay creating an instance until
        # we actually need it. Issue seems to be Python 3.6 specific.
        self._new_executor = functools.partial(
            _get_executor, custom_port_range, executor
        )

        self._ee_con_info: Optional[EvaluatorConnectionInfo] = None
        self._eval_proc: Optional[BaseProcess] = None
        self._iens_to_task = {}  # type: ignore
        self._allow_cancel = multiprocessing.Event()

    @property
    def output_bus(
        self,
    ) -> "asyncio.Queue[CloudEvent]":
        # TODO: the prefect ensemble needs to return the multiprocessing.Queue or keep
        # using websockets. See https://github.com/equinor/ert/issues/3456
        # It cannot use asyncio.Queue as it is unpicklable.
        raise NotImplementedError

    def get_flow(self, iens_range: List[int]) -> Any:
        with Flow(f"Realization range {iens_range}") as flow:
            # one map pr flow (real-batch)
            transmitter_map: _ensemble_transmitter_mapping = {}
            for iens in iens_range:
                transmitter_map[iens] = dict(self._inputs[iens].items())
                for step in self.reals[iens].get_steps_sorted_topologically():
                    inputs = {
                        inp.name: transmitter_map[iens][inp.name] for inp in step.inputs
                    }
                    outputs = self._outputs[iens]
                    # Prefect does not allow retry_delay if max_retries is 0
                    retry_delay = None if self._max_retries == 0 else self._retry_delay
                    assert isinstance(step, (UnixStep, FunctionStep))  # mypy
                    step_task = step.get_task(
                        outputs,
                        self.id_,
                        name=str(iens),
                        max_retries=self._max_retries,
                        retry_delay=retry_delay,
                        on_failure=functools.partial(_on_task_failure, ens_id=self.id_),
                    )
                    result = step_task(inputs=inputs)
                    if iens not in self._iens_to_task:
                        self._iens_to_task[iens] = []
                    self._iens_to_task[iens].append(result)  # type: ignore
                    for output in step.outputs:
                        transmitter_map[iens][output.name] = result[  # type: ignore
                            output.name
                        ]
        return flow

    @staticmethod
    def _get_multiprocessing_context() -> "ForkServerContext":
        """See _prefect_forkserver_preload"""
        preload_module_name = (
            "ert.ensemble_evaluator._builder._prefect_forkserver_preload"
        )
        loader = importlib.util.find_spec(preload_module_name)
        if not loader:
            raise ModuleNotFoundError(f"No module named {preload_module_name}")
        ctx = multiprocessing.get_context("forkserver")
        ctx.set_forkserver_preload([preload_module_name])
        return ctx

    def evaluate(self, config: "EvaluatorServerConfig") -> None:
        self._ee_con_info = config.get_connection_info()

        # everything in self will be pickled since we bind a member function in target
        ctx = self._get_multiprocessing_context()
        eval_proc = ctx.Process(target=self._evaluate)
        eval_proc.daemon = True
        eval_proc.start()
        self._eval_proc = eval_proc
        self._allow_cancel.set()

    def _evaluate(self) -> None:
        get_event_loop()
        assert self._ee_con_info  # mypy
        assert self.id_  # mypy
        try:
            with Client(
                self._ee_con_info.dispatch_uri,
                self._ee_con_info.token,
                self._ee_con_info.cert,
            ) as c:
                event = CloudEvent(
                    {
                        "type": identifiers.EVTYPE_ENSEMBLE_STARTED,
                        "source": f"/ert/ensemble/{self.id_}",
                    },
                )
                c.send(to_json(event).decode())
            with prefect.context(  # type: ignore
                url=self._ee_con_info.dispatch_uri,
                token=self._ee_con_info.token,
                cert=self._ee_con_info.cert,
            ):
                self.run_flow()

            with Client(
                self._ee_con_info.dispatch_uri,
                self._ee_con_info.token,
                self._ee_con_info.cert,
            ) as c:
                event = CloudEvent(
                    {
                        "type": identifiers.EVTYPE_ENSEMBLE_STOPPED,
                        "source": f"/ert/ensemble/{self.id_}",
                        "datacontenttype": "application/octet-stream",
                    },
                    cloudpickle.dumps(self._outputs),
                )
                c.send(to_json(event).decode())
        except Exception as e:  # pylint: disable=broad-except
            logger.exception(
                "An exception occurred while starting the ensemble evaluation",
                exc_info=True,
            )

            # Signal 2 is SIGINT, so it is assumed this exception came from
            # cancellation. This means the ensemble failed event should not be sent.
            if isinstance(e, OSError) and "Signal 2" in str(e):
                logger.debug("interpreting %s as a result of cancellation", e)
                return

            with Client(
                self._ee_con_info.dispatch_uri,
                self._ee_con_info.token,
                self._ee_con_info.cert,
            ) as c:
                event = CloudEvent(
                    {
                        "type": identifiers.EVTYPE_ENSEMBLE_FAILED,
                        "source": f"/ert/ensemble/{self.id_}",
                    },
                )
                c.send(to_json(event).decode())

    def run_flow(self) -> None:
        """Send batches of active realizations as Prefect-flows to the Prefect flow
        executor
        """
        active_iens: List[int] = [realization.iens for realization in self.active_reals]

        i = 0
        state_map = {}
        while i < len(active_iens):
            iens_range = active_iens[i : i + self._real_per_batch]
            flow = self.get_flow(iens_range)
            with prefect_log_level_context(level="WARNING"):
                state = flow.run(executor=self._new_executor())
                if isinstance(state.result, OSError) and "Signal 2" in str(
                    state.result
                ):
                    logger.debug(
                        f"flow failed with {state.result} due to {state.result}"
                    )
                    raise state.result
            for iens in iens_range:
                state_map[iens] = state
            i = i + self._real_per_batch
        for iens, tasks in self._iens_to_task.items():
            for task in tasks:
                if isinstance(state_map[iens].result[task].result, Exception):
                    raise state_map[iens].result[task].result
                for output_name, transmitter in (
                    state_map[iens].result[task].result.items()
                ):
                    self._outputs[iens][output_name] = transmitter

    @property
    def cancellable(self) -> bool:
        return True

    def cancel(self) -> None:
        threading.Thread(target=self._cancel).start()

    def _cancel(self) -> None:
        logger.debug("cancelling, waiting for wakeup...")
        self._allow_cancel.wait()
        logger.debug("got wakeup, killing evaluation process...")

        assert self._ee_con_info  # mypy
        assert self._eval_proc  # mypy
        assert self._eval_proc.pid  # mypy
        os.kill(self._eval_proc.pid, signal.SIGINT)
        start = time.time()
        while self._eval_proc.is_alive() and time.time() - start < 3:
            pass
        if self._eval_proc.is_alive():
            logger.debug("Evaluation process not responding to SIGINT, sending SIGKILL")
            os.kill(self._eval_proc.pid, signal.SIGKILL)
        logger.debug(
            "Evaluation process is %s.",
            "alive" if self._eval_proc.is_alive() else "dead",
        )
        self._eval_proc = None
        event = CloudEvent(
            {
                "type": identifiers.EVTYPE_ENSEMBLE_CANCELLED,
                "source": f"/ert/ensemble/{self.id_}",
                "datacontenttype": "application/json",
            },
        )

        loop = asyncio.new_event_loop()
        loop.run_until_complete(
            self.send_cloudevent(
                self._ee_con_info.dispatch_uri,
                event,
                token=self._ee_con_info.token,
                cert=self._ee_con_info.cert,
            )
        )
        logger.debug("sendt cancelled event")
        loop.close()
