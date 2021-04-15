import asyncio
import contextlib
import logging
import multiprocessing
import os
import signal
import threading
import uuid

import cloudpickle
import prefect.utilities.logging
from cloudevents.http import CloudEvent, to_json
from dask_jobqueue.lsf import LSFJob
from ert_shared.ensemble_evaluator.config import find_open_port
from ert_shared.ensemble_evaluator.entity import identifiers as ids
from ert_shared.ensemble_evaluator.entity.ensemble import (
    _Ensemble,
    create_job_builder,
    create_realization_builder,
    create_step_builder,
)
from ert_shared.ensemble_evaluator.client import Client
from ert_shared.ensemble_evaluator.entity.ensemble import create_file_io_builder
from ert_shared.status.entity import state
from prefect import Flow
from prefect import context as prefect_context
from prefect.executors import DaskExecutor, LocalDaskExecutor

logger = logging.getLogger(__name__)


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


def _get_executor(name="local"):
    if name == "local":
        cluster_kwargs = {
            "silence_logs": "debug",
            "scheduler_options": {"port": find_open_port()},
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
            "scheduler_options": {"port": find_open_port()},
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
    def __init__(self, config):
        self.config = config
        self._ee_dispach_url = config["dispatch_uri"]
        self._reals = self._get_reals()
        self._eval_proc = None
        self._ee_id = None
        self._iens_to_task = {}
        super().__init__(self._reals, metadata={"iter": 0})

    def _get_reals(self):
        real_builders = []
        for iens in range(0, self.config[ids.REALIZATIONS]):
            real_builder = create_realization_builder().active(True).set_iens(iens)
            for step in self.config[ids.STEPS]:
                step_id = uuid.uuid4()
                step_source = f"/ert/ee/{{ee_id}}/real/{iens}/step/{step_id}"
                step_builder = (
                    create_step_builder()
                    .set_id(step_id)
                    .set_name(step[ids.NAME])
                    .set_source(step_source)
                    .set_ee_url(self._ee_dispach_url)
                    .set_type(step[ids.TYPE])
                )

                for io in step.get(ids.INPUTS, []):
                    input_builder = (
                        create_file_io_builder()
                        .set_name(io[ids.RECORD])
                        .set_path(io[ids.LOCATION])
                        .set_mime(io[ids.MIME])
                    )

                    if io.get(ids.IS_EXECUTABLE):
                        input_builder.set_executable()

                    step_builder.add_input(input_builder)
                for io in step.get(ids.OUTPUTS, []):
                    step_builder.add_output(
                        create_file_io_builder()
                        .set_name(io[ids.RECORD])
                        .set_path(io[ids.LOCATION])
                        .set_mime(io[ids.MIME])
                    )

                for job in step[ids.JOBS]:
                    job_builder = (
                        create_job_builder()
                        .set_id(str(uuid.uuid4()))
                        .set_name(job[ids.NAME])
                        .set_executable(job[ids.EXECUTABLE])
                        .set_args(job.get(ids.ARGS))
                        .set_step_source(step_source)
                    )
                    step_builder.add_job(job_builder)
                real_builder.add_step(step_builder)
            real_builders.append(real_builder)
        return [builder.build() for builder in real_builders]

    @staticmethod
    def _on_task_failure(task, state, url):
        if prefect_context.task_run_count > task.max_retries:
            with Client(url) as c:
                event = CloudEvent(
                    {
                        "type": ids.EVTYPE_FM_STEP_FAILURE,
                        "source": task.get_step()._source,
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
                    for record, transmitter in self.config["inputs"][iens].items()
                }
                for step in self._reals[iens].get_steps_sorted_topologically():
                    inputs = {
                        inp.get_name(): transmitter_map[iens][inp.get_name()]
                        for inp in step.get_inputs()
                    }
                    outputs = self.config["outputs"][iens]
                    step_task = step.get_task(outputs, ee_id, name=str(iens))
                    result = step_task(inputs=inputs)
                    self._iens_to_task[iens] = result
                    for output in step.get_outputs():
                        transmitter_map[iens][output.get_name()] = result[
                            output.get_name()
                        ]
        return flow

    def evaluate(self, config, ee_id):
        self._ee_id = ee_id
        mp_ctx = multiprocessing.get_context(method="forkserver")
        self._eval_proc = mp_ctx.Process(
            target=self._evaluate,
            args=(self._ee_dispach_url, ee_id),
        )
        self._eval_proc.daemon = True
        self._eval_proc.start()

    def _evaluate(self, dispatch_url, ee_id):
        asyncio.set_event_loop(asyncio.get_event_loop())
        try:
            with Client(dispatch_url) as c:
                event = CloudEvent(
                    {
                        "type": ids.EVTYPE_ENSEMBLE_STARTED,
                        "source": f"/ert/ee/{self._ee_id}",
                    },
                )
                c.send(to_json(event).decode())
            self.run_flow(ee_id)

            with Client(dispatch_url) as c:
                event = CloudEvent(
                    {
                        "type": ids.EVTYPE_ENSEMBLE_STOPPED,
                        "source": f"/ert/ee/{self._ee_id}",
                        "datacontenttype": "application/octet-stream",
                    },
                    cloudpickle.dumps(self.config["outputs"]),
                )
                c.send(to_json(event).decode())
        except Exception as e:
            logger.exception(
                "An exception occurred while starting the ensemble evaluation",
                exc_info=True,
            )
            with Client(dispatch_url) as c:
                event = CloudEvent(
                    {
                        "type": ids.EVTYPE_ENSEMBLE_FAILED,
                        "source": f"/ert/ee/{self._ee_id}",
                    },
                )
                c.send(to_json(event).decode())

    def run_flow(self, ee_id):
        real_per_batch = self.config[ids.MAX_RUNNING]
        real_range = range(self.config[ids.REALIZATIONS])
        i = 0
        state_map = {}
        while i < self.config[ids.REALIZATIONS]:
            realization_range = real_range[i : i + real_per_batch]
            flow = self.get_flow(ee_id, realization_range)
            with prefect_log_level_context(level="WARNING"):
                state = flow.run(executor=_get_executor(self.config[ids.EXECUTOR]))
            for iens in realization_range:
                state_map[iens] = state
            i = i + real_per_batch
        for iens, task in self._iens_to_task.items():
            for output_name, transmitter in state_map[iens].result[task].result.items():
                self.config["outputs"][iens][output_name] = transmitter

    def is_cancellable(self):
        return True

    def cancel(self):
        threading.Thread(target=self._cancel).start()

    def _cancel(self):
        if self._eval_proc is not None:
            os.kill(self._eval_proc.pid, signal.SIGINT)
        self._eval_proc = None
        event = CloudEvent(
            {
                "type": ids.EVTYPE_ENSEMBLE_CANCELLED,
                "source": f"/ert/ee/{self._ee_id}",
                "datacontenttype": "application/json",
            },
        )

        loop = asyncio.new_event_loop()
        loop.run_until_complete(self.send_cloudevent(self._ee_dispach_url, event))
        loop.close()
