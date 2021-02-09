from pathlib import Path
import os
import os.path
import yaml
import pytest
import asyncio
import websockets
import threading
from datetime import timedelta
from functools import partial
from prefect import Flow
from tests.utils import SOURCE_DIR, tmp
from ert_shared.ensemble_evaluator.config import CONFIG_FILE, load_config
from ert_shared.ensemble_evaluator.evaluator import EnsembleEvaluator
from ert_shared.ensemble_evaluator.prefect_ensemble.prefect_ensemble import (
    PrefectEnsemble,
)
from ert_shared.ensemble_evaluator.prefect_ensemble.unix_step import UnixStep
from ert_shared.ensemble_evaluator.prefect_ensemble.storage_driver import (
    storage_driver_factory,
)
from ert_shared.ensemble_evaluator.entity import identifiers as ids
from ert_shared.ensemble_evaluator.prefect_ensemble.client import Client


def parse_config(path):
    conf_path = Path(path).resolve()
    with open(conf_path, "r") as f:
        return yaml.safe_load(f)


@pytest.mark.timeout(60)
def test_run_prefect_ensemble(unused_tcp_port):
    with tmp(Path(SOURCE_DIR) / "test-data/local/prefect_test_case"):
        conf_file = Path(CONFIG_FILE)
        config = parse_config("config.yml")
        config.update({"config_path": os.getcwd()})
        config.update({"realizations": 2})
        config.update({"executor": "local"})

        with open(conf_file, "w") as f:
            f.write(f'port: "{unused_tcp_port}"\n')

        service_config = load_config(conf_file)
        config.update(service_config)
        ensemble = PrefectEnsemble(config)

        evaluator = EnsembleEvaluator(ensemble, config, 0, ee_id="1")

        mon = evaluator.run()

        for event in mon.track():
            if event.data is not None and event.data.get("status") == "Stopped":
                mon.signal_done()

        assert evaluator._snapshot.get_status() == "Stopped"

        successful_realizations = evaluator._snapshot.get_successful_realizations()

        assert successful_realizations == config["realizations"]


@pytest.mark.timeout(60)
def test_run_prefect_ensemble_with_path(unused_tcp_port):
    with tmp(os.path.join(SOURCE_DIR, "test-data/local/prefect_test_case")):
        conf_file = Path(CONFIG_FILE)
        config = parse_config("config.yml")
        config.update({"config_path": Path.cwd()})
        config.update({"realizations": 2})
        config.update({"executor": "local"})

        config["config_path"] = Path(config["config_path"])
        config["run_path"] = Path(config["run_path"])
        config["storage"]["storage_path"] = Path(config["storage"]["storage_path"])

        with open(conf_file, "w") as f:
            f.write(f'port: "{unused_tcp_port}"\n')

        service_config = load_config(conf_file)
        config.update(service_config)
        ensemble = PrefectEnsemble(config)

        evaluator = EnsembleEvaluator(ensemble, config, 0, ee_id="1")

        mon = evaluator.run()

        for event in mon.track():
            if event.data is not None and event.data.get("status") == "Stopped":
                mon.signal_done()

        assert evaluator._snapshot.get_status() == "Stopped"

        successful_realizations = evaluator._snapshot.get_successful_realizations()

        assert successful_realizations == config["realizations"]


@pytest.mark.timeout(60)
def test_cancel_run_prefect_ensemble(unused_tcp_port):
    with tmp(Path(SOURCE_DIR) / "test-data/local/prefect_test_case"):
        conf_file = Path(CONFIG_FILE)
        config = parse_config("config.yml")
        config.update({"config_path": Path.absolute(Path("."))})
        config.update({"realizations": 2})
        config.update({"executor": "local"})

        with open(conf_file, "w") as f:
            f.write(f'port: "{unused_tcp_port}"\n')

        service_config = load_config(conf_file)
        config.update(service_config)
        ensemble = PrefectEnsemble(config)

        evaluator = EnsembleEvaluator(ensemble, config, 0, ee_id="2")

        mon = evaluator.run()
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

        stage_task = UnixStep(
            resources=[resource],
            outputs=["output.out"],
            job_list=jobs,
            iens=1,
            cmd="python3",
            url=url,
            step_id="step_id_0",
            stage_id="stage_id_0",
            ee_id="ee_id_0",
            on_failure=_on_task_failure,
            run_path=config.get("run_path"),
            storage_config=config.get("storage"),
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

        stage_task = UnixStep(
            resources=[resource],
            outputs=["output.out"],
            job_list=jobs,
            iens=1,
            cmd="python3",
            url=url,
            step_id="step_id_0",
            stage_id="stage_id_0",
            ee_id="ee_id_0",
            run_path=config.get("run_path"),
            storage_config=config.get("storage"),
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

        stage_task = UnixStep(
            resources=[resource],
            outputs=[],
            job_list=jobs,
            iens=1,
            cmd="python3",
            url=url,
            step_id="step_id_0",
            stage_id="stage_id_0",
            ee_id="ee_id_0",
            on_failure=partial(PrefectEnsemble._on_task_failure, url=url),
            run_path=config.get("run_path"),
            storage_config=config.get("storage"),
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
