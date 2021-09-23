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
from datetime import timedelta
from multiprocessing.context import BaseContext
from typing import Optional

import cloudpickle
import prefect
import prefect.utilities.logging
from cloudevents.http import CloudEvent, to_json
from dask_jobqueue.lsf import LSFJob
from prefect import Flow
from prefect import context as prefect_context
from prefect.executors import DaskExecutor, LocalDaskExecutor

from ert_shared.ensemble_evaluator.client import Client
from ert_shared.ensemble_evaluator.config import EvaluatorServerConfig
from ert_shared.ensemble_evaluator.ensemble.base import _Ensemble
from ert_shared.ensemble_evaluator.entity import identifiers as ids
from ert_shared.port_handler import find_available_port

logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES = 0
DEFAULT_RETRY_DELAY = 5  # seconds


@contextlib.contextmanager
def prefect_log_level_context(level):
    prefect_logger = prefect.utilities.logging.get_logger()
    prev_log_level = prefect_logger.level
    prefect_logger.setLevel(level=level)
    yield
    prefect_logger.setLevel(level=prev_log_level)


async def _eq_submit_job(self, script_filename):
    with open(script_filename) as fh:
        lines = fh.readlines()[1:]
    lines = [
        line.strip() if "#BSUB" not in line else line[5:].strip() for line in lines
    ]
    piped_cmd = [self.submit_command + " ".join(lines)]
    return self._call(piped_cmd, shell=True)


def _get_executor(custom_port_range, name="local"):
    _, port = find_available_port(custom_range=custom_port_range)
    if name == "local":
        cluster_kwargs = {
            "silence_logs": "debug",
            "scheduler_options": {"port": port},
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
            "cores": 4,
            "memory": "16GB",
            "resource_spec": "select=1:ncpus=4:mem=16GB",
        }
        return DaskExecutor(
            cluster_class="dask_jobqueue.PBSCluster",
            cluster_kwargs=cluster_kwargs,
            debug=True,
        )
    else:
        raise ValueError(f"Unknown executor name {name}")


class PrefectEnsemble(_Ensemble):
    def __init__(
        self,
        reals,
        inputs,
        outputs,
        max_running,
        max_retries,
        executor,
        retry_delay,
        custom_port_range=None,
    ):
        super().__init__(reals=reals, metadata={"iter": 0})
        self._inputs = inputs
        self._outputs = outputs
        self._real_per_batch = max_running

        self._max_retries = max_retries
        self._retry_delay = timedelta(seconds=retry_delay)

        # If we instansiate an executor here the prefect ensemble
        # will fail to pickle (required when using multiprocessing),
        # as the executor has an internal thread lock. Hence, we
        # bind the parameters and delay creating an instance until
        # we actually need it. Issue seems to be Python 3.6 specific.
        self._new_executor = functools.partial(
            _get_executor, custom_port_range, executor
        )

        self._ee_config = None
        self._eval_proc = None
        self._ee_id: Optional[str] = None
        self._iens_to_task = {}
        self._allow_cancel = multiprocessing.Event()

    def _on_task_failure(self, task, state):
        if prefect_context.task_run_count > task.max_retries:
            url = prefect_context.url
            token = prefect_context.token
            cert = prefect_context.cert
            with Client(url, token, cert) as c:
                event = CloudEvent(
                    {
                        "type": ids.EVTYPE_FM_STEP_FAILURE,
                        "source": task.get_step().get_source(self._ee_id),
                        "datacontenttype": "application/json",
                    },
                    {"error_msg": state.message},
                )
                c.send(to_json(event).decode())

    def get_flow(self, ee_id, real_range):
        with Flow(f"Realization range {real_range}") as flow:
            transmitter_map = {}
            for iens in real_range:
                transmitter_map[iens] = {
                    record: transmitter
                    for record, transmitter in self._inputs[iens].items()
                }
                for step in self._reals[iens].get_steps_sorted_topologically():
                    inputs = {
                        inp.get_name(): transmitter_map[iens][inp.get_name()]
                        for inp in step.get_inputs()
                    }
                    outputs = self._outputs[iens]
                    # Prefect does not allow retry_delay if max_retries is 0
                    retry_delay = None if self._max_retries == 0 else self._retry_delay
                    step_task = step.get_task(
                        outputs,
                        ee_id,
                        name=str(iens),
                        max_retries=self._max_retries,
                        retry_delay=retry_delay,
                        on_failure=self._on_task_failure,
                    )
                    result = step_task(inputs=inputs)
                    if iens not in self._iens_to_task:
                        self._iens_to_task[iens] = []
                    self._iens_to_task[iens].append(result)
                    for output in step.get_outputs():
                        transmitter_map[iens][output.get_name()] = result[
                            output.get_name()
                        ]
        return flow

    @staticmethod
    def _get_multiprocessing_context() -> BaseContext:
        """See _prefect_forkserver_preload"""
        preload_module_name = (
            "ert_shared.ensemble_evaluator.ensemble._prefect_forkserver_preload"
        )
        loader = importlib.util.find_spec(preload_module_name)
        if not loader:
            raise ModuleNotFoundError(f"No module named {preload_module_name}")
        ctx = multiprocessing.get_context("forkserver")
        ctx.set_forkserver_preload([preload_module_name])
        return ctx

    def evaluate(self, config: EvaluatorServerConfig, ee_id: str):
        self._ee_id = ee_id
        self._ee_config = config

        ctx = self._get_multiprocessing_context()
        self._eval_proc = ctx.Process(
            target=self._evaluate,
            args=(config, ee_id),
        )
        self._eval_proc.daemon = True
        self._eval_proc.start()
        self._allow_cancel.set()

    def _evaluate(self, ee_config: EvaluatorServerConfig, ee_id):
        asyncio.set_event_loop(asyncio.get_event_loop())
        try:
            with Client(ee_config.dispatch_uri, ee_config.token, ee_config.cert) as c:
                event = CloudEvent(
                    {
                        "type": ids.EVTYPE_ENSEMBLE_STARTED,
                        "source": f"/ert/ee/{self._ee_id}",
                    },
                )
                c.send(to_json(event).decode())
            with prefect.context(
                url=ee_config.dispatch_uri, token=ee_config.token, cert=ee_config.cert
            ):
                self.run_flow(ee_id)

            with Client(ee_config.dispatch_uri, ee_config.token, ee_config.cert) as c:
                event = CloudEvent(
                    {
                        "type": ids.EVTYPE_ENSEMBLE_STOPPED,
                        "source": f"/ert/ee/{self._ee_id}",
                        "datacontenttype": "application/octet-stream",
                    },
                    cloudpickle.dumps(self._outputs),
                )
                c.send(to_json(event).decode())
        except Exception as e:
            logger.exception(
                "An exception occurred while starting the ensemble evaluation",
                exc_info=True,
            )
            with Client(ee_config.dispatch_uri, ee_config.token, ee_config.cert) as c:
                event = CloudEvent(
                    {
                        "type": ids.EVTYPE_ENSEMBLE_FAILED,
                        "source": f"/ert/ee/{self._ee_id}",
                    },
                )
                c.send(to_json(event).decode())

    def run_flow(self, ee_id):
        num_realizations = len(self._reals)
        real_range = range(num_realizations)
        i = 0
        state_map = {}
        while i < num_realizations:
            realization_range = real_range[i : i + self._real_per_batch]
            flow = self.get_flow(ee_id, realization_range)
            with prefect_log_level_context(level="WARNING"):
                state = flow.run(executor=self._new_executor())
            for iens in realization_range:
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

    def is_cancellable(self):
        return True

    def cancel(self):
        threading.Thread(target=self._cancel).start()

    def _cancel(self):
        logger.debug("cancelling, waiting for wakeup...")
        self._allow_cancel.wait()
        logger.debug("got wakeup, killing evaluation process...")

        if self._eval_proc is not None:
            os.kill(self._eval_proc.pid, signal.SIGINT)
            start = time.time()
            while self._eval_proc.is_alive() and time.time() - start < 3:
                pass
            if self._eval_proc.is_alive():
                logger.debug(
                    "Evaluation process is not responding to SIGINT, escalating to SIGKILL"
                )
                os.kill(self._eval_proc.pid, signal.SIGKILL)

        self._eval_proc = None
        event = CloudEvent(
            {
                "type": ids.EVTYPE_ENSEMBLE_CANCELLED,
                "source": f"/ert/ee/{self._ee_id}",
                "datacontenttype": "application/json",
            },
        )

        loop = asyncio.new_event_loop()
        loop.run_until_complete(
            self.send_cloudevent(
                self._ee_config.dispatch_uri,
                event,
                token=self._ee_config.token,
                cert=self._ee_config.cert,
            )
        )
        loop.close()
