from pathlib import Path
import sys
import os
import os.path
import yaml
import pytest
import asyncio
import websockets
import threading
import json
from datetime import timedelta
from functools import partial
from unittest.mock import patch
from prefect import Flow
from tests.utils import SOURCE_DIR, tmp
from ert_shared.ensemble_evaluator.config import EvaluatorServerConfig
from ert_shared.ensemble_evaluator.evaluator import EnsembleEvaluator
from ert_shared.ensemble_evaluator.prefect_ensemble.prefect_ensemble import (
    PrefectEnsemble,
)
from ert_shared.ensemble_evaluator.prefect_ensemble.unix_step import UnixStep
from ert_shared.ensemble_evaluator.prefect_ensemble.function_step import FunctionStep
from ert_shared.ensemble_evaluator.prefect_ensemble.storage_driver import (
    storage_driver_factory,
)
from ert_shared.ensemble_evaluator.entity import identifiers as ids
from ert_shared.ensemble_evaluator.prefect_ensemble.client import Client


def parse_config(path):
    conf_path = Path(path).resolve()
    with open(conf_path, "r") as f:
        return yaml.safe_load(f)


def input_files(config, coefficients):
    paths = {}
    storage = storage_driver_factory(config.get("storage"), ".")
    file_name = "coeffs.json"
    for iens, values in enumerate(coefficients):
        with open(file_name, "w") as f:
            json.dump(values, f)
        paths[iens] = (storage.store(file_name, iens),)
    return paths


@pytest.fixture()
def coefficients():
    return [{"a": a, "b": b, "c": c} for (a, b, c) in [(1, 2, 3), (4, 2, 1)]]


@pytest.mark.timeout(60)
def test_run_prefect_ensemble(unused_tcp_port, coefficients):
    with tmp(Path(SOURCE_DIR) / "test-data/local/prefect_test_case"):
        config = parse_config("config.yml")
        config.update(
            {
                "config_path": os.getcwd(),
                "realizations": 2,
                "executor": "local",
                "input_files": input_files(config, coefficients),
            }
        )

        service_config = EvaluatorServerConfig(unused_tcp_port)
        ensemble = PrefectEnsemble(config)

        evaluator = EnsembleEvaluator(ensemble, service_config, 0, ee_id="1")

        with evaluator.run() as mon:
            for event in mon.track():
                if event.data is not None and event.data.get("status") in [
                    "Failed",
                    "Stopped",
                ]:
                    mon.signal_done()

        assert evaluator._snapshot.get_status() == "Stopped"

        successful_realizations = evaluator._snapshot.get_successful_realizations()

        assert successful_realizations == config["realizations"]


@pytest.mark.timeout(60)
def test_run_prefect_ensemble_with_path(unused_tcp_port, coefficients):
    with tmp(os.path.join(SOURCE_DIR, "test-data/local/prefect_test_case")):
        config = parse_config("config.yml")
        config.update(
            {
                "config_path": Path.cwd(),
                "realizations": 2,
                "executor": "local",
                "input_files": input_files(config, coefficients),
            }
        )

        config["config_path"] = Path(config["config_path"])
        config["run_path"] = Path(config["run_path"])
        config["storage"]["storage_path"] = Path(config["storage"]["storage_path"])

        service_config = EvaluatorServerConfig(unused_tcp_port)
        ensemble = PrefectEnsemble(config)

        evaluator = EnsembleEvaluator(ensemble, service_config, 0, ee_id="1")

        with evaluator.run() as mon:
            for event in mon.track():
                if event.data is not None and event.data.get("status") in [
                    "Failed",
                    "Stopped",
                ]:
                    mon.signal_done()

        assert evaluator._snapshot.get_status() == "Stopped"

        successful_realizations = evaluator._snapshot.get_successful_realizations()

        assert successful_realizations == config["realizations"]


@pytest.mark.timeout(60)
def test_cancel_run_prefect_ensemble(unused_tcp_port, coefficients):
    with tmp(Path(SOURCE_DIR) / "test-data/local/prefect_test_case"):
        config = parse_config("config.yml")
        config.update(
            {
                "config_path": os.getcwd(),
                "realizations": 2,
                "executor": "local",
                "input_files": input_files(config, coefficients),
            }
        )

        service_config = EvaluatorServerConfig(unused_tcp_port)
        ensemble = PrefectEnsemble(config)

        evaluator = EnsembleEvaluator(ensemble, service_config, 0, ee_id="2")

        with evaluator.run() as mon:
            cancel = True
            for _ in mon.track():
                if cancel:
                    mon.signal_cancel()
                    cancel = False

        assert evaluator._snapshot.get_status() == "Cancelled"


def _mock_ws(host, port, messages):
    loop = asyncio.new_event_loop()
    done = loop.create_future()

    async def _handler(websocket, path):
        while True:
            msg = await websocket.recv()
            messages.append(msg)
            if msg == "stop":
                done.set_result(None)
                break

    async def _run_server():
        async with websockets.serve(_handler, host, port):
            await done

    loop.run_until_complete(_run_server())
    loop.close()


def test_prefect_client(unused_tcp_port):
    host = "localhost"
    url = f"ws://{host}:{unused_tcp_port}"
    messages = []
    mock_ws_thread = threading.Thread(
        target=partial(_mock_ws, messages=messages), args=(host, unused_tcp_port)
    )

    mock_ws_thread.start()
    messages_c1 = ["test_1", "test_2", "test_3", "stop"]

    with Client(url) as c1:
        for msg in messages_c1:
            c1.send(msg)

    with Client(url, max_retries=2, timeout_multiplier=1) as c2:
        c2.send("after_ws_stoped")

    mock_ws_thread.join()

    for msg in messages_c1:
        assert msg in messages

    assert "after_ws_stoped" not in messages


def test_unix_step(unused_tcp_port):
    host = "localhost"
    url = f"ws://{host}:{unused_tcp_port}"
    messages = []
    mock_ws_thread = threading.Thread(
        target=partial(_mock_ws, messages=messages), args=(host, unused_tcp_port)
    )

    mock_ws_thread.start()

    def _on_task_failure(task, state):
        raise Exception(state.message)

    with tmp(Path(SOURCE_DIR) / "test-data/local/prefect_test_case"):
        config = parse_config("config.yml")
        storage = storage_driver_factory(config=config.get("storage"), run_path=".")
        resource = storage.store("unix_test_script.py")
        jobs = [
            {
                "id": "0",
                "name": "test_script",
                "executable": "unix_test_script.py",
                "args": ["vas"],
            }
        ]
        step = {
            "outputs": ["output.out"],
            "iens": 1,
            "step_id": "step_id_0",
            "stage_id": "stage_id_0",
            "jobs": jobs,
        }

        stage_task = UnixStep(
            step=step,
            resources=[resource],
            cmd="python3",
            url=url,
            ee_id="ee_id_0",
            on_failure=_on_task_failure,
            run_path=config.get(ids.RUN_PATH),
            storage_config=config.get(ids.STORAGE),
            max_retries=1,
            retry_delay=timedelta(seconds=2),
        )

        flow = Flow("testing")
        flow.add_task(stage_task)
        flow_run = flow.run()

        # Stop the mock evaluator WS server
        with Client(url) as c:
            c.send("stop")
        mock_ws_thread.join()

        task_result = flow_run.result[stage_task]
        assert task_result.is_successful()
        assert flow_run.is_successful()

        assert len(task_result.result["outputs"]) == 1
        expected_path = storage.get_storage_path(1) / "output.out"
        output_path = flow_run.result[stage_task].result["outputs"][0]
        assert expected_path == output_path
        assert output_path.exists()


def test_function_step(unused_tcp_port):
    host = "localhost"
    url = f"ws://{host}:{unused_tcp_port}"
    messages = []
    mock_ws_thread = threading.Thread(
        target=partial(_mock_ws, messages=messages), args=(host, unused_tcp_port)
    )

    mock_ws_thread.start()

    def _on_task_failure(task, state):
        raise Exception(state.message)

    with tmp(Path(SOURCE_DIR) / "test-data/local/prefect_test_case"):
        config = parse_config("config.yml")
        storage = storage_driver_factory(config=config.get("storage"), run_path=".")

        def sum_function(values):
            return sum(values)

        jobs = [
            {
                "id": "0",
                "name": "test_script",
                "executable": sum_function,
                "output": "output.out",
            }
        ]
        test_values = {"values": [42, 24, 6]}
        step = {
            "jobs": jobs,
            "step_id": "step_id_0",
            "stage_id": "stage_id_0",
            "iens": 1,
            "step_input": test_values,
        }

        function_task = FunctionStep(
            step=step,
            url=url,
            ee_id="ee_id_0",
            on_failure=_on_task_failure,
            storage_config=config.get("storage"),
            max_retries=1,
            retry_delay=timedelta(seconds=2),
        )

        flow = Flow("testing")
        flow.add_task(function_task)
        flow_run = flow.run()

        # Stop the mock evaluator WS server
        with Client(url) as c:
            c.send("stop")
        mock_ws_thread.join()

        task_result = flow_run.result[function_task]
        assert task_result.is_successful()
        assert flow_run.is_successful()

        assert len(task_result.result["outputs"]) == 1
        expected_path = storage.get_storage_path(1) / "output.out"
        output_path = flow_run.result[function_task].result["outputs"][0]
        assert expected_path == output_path
        assert output_path.exists()
        with open(output_path, "r") as f:
            result = json.load(f)
        assert sum_function(**test_values) == result


def test_unix_step_error(unused_tcp_port):
    host = "localhost"
    url = f"ws://{host}:{unused_tcp_port}"
    messages = []
    mock_ws_thread = threading.Thread(
        target=partial(_mock_ws, messages=messages), args=(host, unused_tcp_port)
    )

    mock_ws_thread.start()

    def _on_task_failure(task, state):
        raise Exception(state.message)

    with tmp(Path(SOURCE_DIR) / "test-data/local/prefect_test_case", False):
        config = parse_config("config.yml")
        storage = storage_driver_factory(config=config.get("storage"), run_path=".")
        resource = storage.store("unix_test_script.py")
        jobs = [
            {
                "id": "0",
                "name": "test_script",
                "executable": "unix_test_script.py",
                "args": ["foo", "bar"],
            }
        ]
        step = {
            "outputs": ["output.out"],
            "iens": 1,
            "step_id": "step_id_0",
            "stage_id": "stage_id_0",
            "jobs": jobs,
        }

        stage_task = UnixStep(
            step=step,
            resources=[resource],
            cmd="python3",
            url=url,
            ee_id="ee_id_0",
            on_failure=_on_task_failure,
            run_path=config.get(ids.RUN_PATH),
            storage_config=config.get(ids.STORAGE),
            max_retries=1,
            retry_delay=timedelta(seconds=2),
        )

        flow = Flow("testing")
        flow.add_task(stage_task)
        flow_run = flow.run()

        # Stop the mock evaluator WS server
        with Client(url) as c:
            c.send("stop")
        mock_ws_thread.join()

        task_result = flow_run.result[stage_task]
        assert not task_result.is_successful()
        assert not flow_run.is_successful()

        assert isinstance(task_result.result, Exception)
        assert (
            "Script test_script failed with exception usage: unix_test_script.py [-h] argument"
            in task_result.message
        )


def test_on_task_failure(unused_tcp_port):
    host = "localhost"
    url = f"ws://{host}:{unused_tcp_port}"
    messages = []
    mock_ws_thread = threading.Thread(
        target=partial(_mock_ws, messages=messages), args=(host, unused_tcp_port)
    )

    mock_ws_thread.start()

    with tmp(Path(SOURCE_DIR) / "test-data/local/prefect_test_case", False):
        config = parse_config("config.yml")
        storage = storage_driver_factory(config=config.get("storage"), run_path=".")
        resource = storage.store("unix_test_retry_script.py")
        jobs = [
            {
                "id": "0",
                "name": "test_script",
                "executable": "unix_test_retry_script.py",
                "args": [],
            }
        ]
        step = {
            "outputs": [],
            "iens": 1,
            "step_id": "step_id_0",
            "stage_id": "stage_id_0",
            "jobs": jobs,
        }

        stage_task = UnixStep(
            step=step,
            resources=[resource],
            cmd="python3",
            url=url,
            ee_id="ee_id_0",
            on_failure=partial(PrefectEnsemble._on_task_failure, url=url),
            run_path=config.get(ids.RUN_PATH),
            storage_config=config.get(ids.STORAGE),
            max_retries=3,
            retry_delay=timedelta(seconds=1),
        )

        flow = Flow("testing")
        flow.add_task(stage_task)
        flow_run = flow.run()

        # Stop the mock evaluator WS server
        with Client(url) as c:
            c.send("stop")
        mock_ws_thread.join()

        task_result = flow_run.result[stage_task]
        assert task_result.is_successful()
        assert flow_run.is_successful()

        fail_job_messages = [
            msg for msg in messages if ids.EVTYPE_FM_JOB_FAILURE in msg
        ]
        fail_step_messages = [
            msg for msg in messages if ids.EVTYPE_FM_STEP_FAILURE in msg
        ]

        expected_job_failed_messages = 2
        expected_step_failed_messages = 0
        assert expected_job_failed_messages == len(fail_job_messages)
        assert expected_step_failed_messages == len(fail_step_messages)


def dummy_get_flow(*args, **kwargs):
    raise RuntimeError()


@pytest.mark.timeout(60)
def test_run_prefect_ensemble_exception(unused_tcp_port, coefficients):
    with tmp(os.path.join(SOURCE_DIR, "test-data/local/prefect_test_case")):
        config = parse_config("config.yml")
        config.update(
            {
                "config_path": os.getcwd(),
                "realizations": 2,
                "executor": "local",
                "input_files": input_files(config, coefficients),
            }
        )

        service_config = EvaluatorServerConfig(unused_tcp_port)

        ensemble = PrefectEnsemble(config)
        evaluator = EnsembleEvaluator(ensemble, service_config, 0, ee_id="1")

        ensemble.get_flow = dummy_get_flow
        with evaluator.run() as mon:
            for event in mon.track():
                if event.data is not None and event.data.get("status") in [
                    "Failed",
                    "Stopped",
                ]:
                    mon.signal_done()
        assert evaluator._snapshot.get_status() == "Failed"
