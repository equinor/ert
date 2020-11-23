import asyncio
import os
import socket
import threading

import cloudevents
import prefect
import websockets
from dask_jobqueue.lsf import LSFJob
from ert_shared.ensemble_evaluator.entity import identifiers as ids
from ert_shared.ensemble_evaluator.entity.ensemble import (
    _Ensemble,
    create_ensemble_builder,
    create_function_job_builder,
    create_realization_builder,
    create_stage_builder,
    create_step_builder,
)
from ert_shared.ensemble_evaluator.ws_util import wait_for_ws


def get_ip_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    return s.getsockname()[0]


class _EventTranslator:
    _queue_state_to_stage_event_type_map = {
        "Pending": ids.EVTYPE_FM_STAGE_WAITING,
        "Running": ids.EVTYPE_FM_STAGE_RUNNING,
        "Success": ids.EVTYPE_FM_STAGE_SUCCESS,
        "Failed": ids.EVTYPE_FM_STAGE_FAILURE,
    }

    _queue_state_to_legacy_queue_state_map = {
        "Pending": "JOB_QUEUE_WAITING",
        "Running": "JOB_QUEUE_RUNNING",
        "Success": "JOB_QUEUE_SUCCESS",
        "Failed": "JOB_QUEUE_FAILED",
    }

    _queue_state_to_step_event_type_map = {
        "Running": ids.EVTYPE_FM_STEP_START,
        "Success": ids.EVTYPE_FM_STEP_SUCCESS,
        "Failed": ids.EVTYPE_FM_STEP_FAILURE,
    }

    _queue_state_to_job_event_type_map = {
        "Running": ids.EVTYPE_FM_JOB_RUNNING,
        "Success": ids.EVTYPE_FM_JOB_SUCCESS,
        "Failed": ids.EVTYPE_FM_JOB_FAILURE,
    }

    @staticmethod
    def _task_state_to_stage_event_type(state):
        return _EventTranslator._queue_state_to_stage_event_type_map.get(
            state, ids.EVTYPE_FM_STAGE_UNKNOWN
        )

    @staticmethod
    def _task_state_to_legacy_type(state):
        return _EventTranslator._queue_state_to_legacy_queue_state_map.get(
            state, "JOB_QUEUE_UNKNOWN"
        )

    @staticmethod
    def _task_state_to_step_event_type(state):
        return _EventTranslator._queue_state_to_step_event_type_map[state]

    @staticmethod
    def _task_state_to_job_event_type(state):
        return _EventTranslator._queue_state_to_job_event_type_map[state]

    @staticmethod
    def create_stage_event(state, real_id):
        return cloudevents.http.CloudEvent(
            {
                "type": _EventTranslator._task_state_to_stage_event_type(state),
                "source": f"/ert/ee/{0}/real/{real_id}/stage/{0}",
                "datacontenttype": "application/json",
            },
            {
                "queue_event_type": _EventTranslator._task_state_to_legacy_type(state),
            },
        )

    @staticmethod
    def create_step_event(state, real_id):
        return cloudevents.http.CloudEvent(
            {
                "type": _EventTranslator._task_state_to_step_event_type(state),
                "source": f"/ert/ee/{0}/real/{real_id}/stage/{0}/step/{0}",
                "datacontenttype": "application/json",
            },
            {
                "jobs": "dummy",
            },
        )

    @staticmethod
    def create_job_event(state, real_id):
        return cloudevents.http.CloudEvent(
            {
                "type": _EventTranslator._task_state_to_job_event_type(state),
                "source": f"/ert/ee/{0}/real/{real_id}/stage/{0}/step/{0}/job/{0}",
                "datacontenttype": "application/json",
            },
            {
                "jobs": "dummy",
            },
        )


class _PrefectEventHandler:
    def __init__(self, url):
        self._url = url

    @staticmethod
    def _get_real_id_from_slug(task):
        slug = task.slug
        if slug is None:
            return None
        suffix = "-copy"
        if slug.endswith(suffix):
            slug = slug[: -len(suffix)]
        try:
            return int(slug.split("-")[-1])
        except ValueError:
            return None

    def _handle_flow_state(self, task, old_state, new_state):
        # TODO: Ensemble level events
        pass

    def _handle_task_state(self, task, old_state, new_state):
        logger = prefect.context.get("logger")

        real_id = self._get_real_id_from_slug(task)
        if real_id is None:
            logger.warning(f"Could not get real id from {task}.")
            return

        if isinstance(new_state, prefect.engine.state.Pending):
            self._send_one_event(
                _EventTranslator.create_stage_event(state="Pending", real_id=real_id)
            )
        elif isinstance(new_state, prefect.engine.state.Running):
            self._publish_state_for_all(state="Running", real_id=real_id)
        elif isinstance(new_state, prefect.engine.state.Success):
            self._publish_state_for_all(state="Success", real_id=real_id)
        elif isinstance(new_state, prefect.engine.state.Failed):
            self._publish_state_for_all(state="Failed", real_id=real_id)
        else:
            self._send_one_event(
                _EventTranslator.create_stage_event(state="Unknown", real_id=real_id)
            )
            logger.warning(f"Unknown state {new_state}.")

    def ert_state_handler(self, task, old_state, new_state):
        if isinstance(task, prefect.Flow):
            self._handle_flow_state(task, old_state, new_state)
        if isinstance(task, prefect.Task):
            self._handle_task_state(task, old_state, new_state)
        return new_state

    def _send_one_event(self, event):
        self._send_events([event])

    def _send_events(self, events):
        async def send():
            async with websockets.connect(self._url) as websocket:
                for event in events:
                    if event is None:
                        await websocket.send("null")
                        return
                    await websocket.send(cloudevents.http.to_json(event))

        asyncio.get_event_loop().run_until_complete(send())

    def _publish_state_for_all(self, state, real_id):
        self._send_events(
            [
                _EventTranslator.create_stage_event(state=state, real_id=real_id),
                _EventTranslator.create_step_event(state=state, real_id=real_id),
                _EventTranslator.create_job_event(state=state, real_id=real_id),
            ],
        )


class _FunctionTask(prefect.Task):
    def __init__(self, fun=lambda: None, **kwargs):
        super().__init__(**kwargs)
        self._fun = fun

    def run(self, func_input):
        return self._fun(**func_input)


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
        return prefect.engine.executors.DaskExecutor()
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
        }
        return prefect.engine.executors.DaskExecutor(
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
        return prefect.engine.executors.DaskExecutor(
            cluster_class="dask_jobqueue.PBSCluster",
            cluster_kwargs=cluster_kwargs,
            debug=True,
        )
    else:
        raise ValueError(f"Unknown executor name {name}")


class _FunctionEnsemble(_Ensemble):
    def __init__(
        self,
        fun=lambda x: None,
        inputs={},
        executor="local",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._fun = fun
        self._inputs = inputs
        self._executor = executor

        self._evaluation_thread = None
        self._evaluation_result = None

    def _set_all_pending(self, url, real_ids):
        event_handler = _PrefectEventHandler(url=url)
        event_handler._send_events(
            [
                _EventTranslator.create_stage_event(state="Pending", real_id=real_id)
                for real_id in real_ids
            ]
        )

    def evaluate(self, host, port):
        host = get_ip_address()
        self._evaluation_thread = threading.Thread(
            target=self._evaluate,
            args=(
                self._inputs,
                asyncio.get_event_loop(),
                f"ws://{host}:{port}/dispatch",
            ),
        )
        self._evaluation_thread.start()

    def _evaluate(self, inputs, loop, url):
        asyncio.set_event_loop(loop)

        inputs = {real_id: func_input for real_id, func_input in enumerate(inputs)}
        outputs = {}

        with prefect.Flow(
            "ERT3 Flow",
            state_handlers=[_PrefectEventHandler(url=url).ert_state_handler],
        ) as flow:
            for real_id, func_input in inputs.items():
                task = _FunctionTask(
                    fun=self._fun,
                    state_handlers=[_PrefectEventHandler(url=url).ert_state_handler],
                    slug=f"function-task-{real_id}",
                )
                outputs[real_id] = task(func_input)

        self._set_all_pending(url=url, real_ids=inputs.keys())
        state = flow.run(executor=_get_executor(self._executor))

        results = [None] * len(inputs)
        for real_id in inputs.keys():
            results[real_id] = state.result[outputs[real_id]].result

        self._evaluation_result = results

    def join(self):
        self._evaluation_thread.join()
        if "DASK_PARENT" in os.environ:
            os.environ.pop("DASK_PARENT")

    def results(self):
        self.join()
        return self._evaluation_result


def create_function_ensemble(fun, inputs, executor="local"):
    builder = create_ensemble_builder()

    for iens in range(0, len(inputs)):
        step = create_step_builder().set_id(0)
        step.add_job(
            create_function_job_builder().set_function(fun).set_id(0)
        ).set_dummy_io()

        builder.add_realization(
            create_realization_builder()
            .add_stage(
                create_stage_builder().add_step(step).set_id(0).set_status("Unknown")
            )
            .set_iens(iens)
            .active(True)
        )
    return builder.build(
        ensemble_cls=_FunctionEnsemble,
        fun=fun,
        inputs=inputs,
        executor=executor,
    )
