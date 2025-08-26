import asyncio
import logging
import os
import ssl
from base64 import b64encode
from dataclasses import dataclass
from pathlib import Path
from shutil import which
from unittest.mock import MagicMock, patch

import pytest
from fastapi.encoders import jsonable_encoder
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import ert
from ert.dark_storage.app import app
from ert.ensemble_evaluator import EndEvent
from ert.scheduler.event import FinishedEvent
from everest.config import EverestConfig, ServerConfig
from everest.detached import (
    ExperimentState,
    everserver,
    everserver_status,
    start_experiment,
    start_server,
    wait_for_server,
)
from everest.everest_storage import EverestStorage
from everest.strings import OPT_FAILURE_ALL_REALIZATIONS, OPT_FAILURE_REALIZATIONS


@pytest.fixture
def setup_client(monkeypatch):
    def func(events=None):
        events = [EndEvent(failed=False, msg="Complete")] if events is None else events
        subscribers = {}
        server_config_mock = MagicMock()
        monkeypatch.setattr(
            ert.dark_storage.endpoints.experiment_server.shared_data, "events", events
        )
        monkeypatch.setattr(
            ert.dark_storage.endpoints.experiment_server.shared_data,
            "subscribers",
            subscribers,
        )

        def getitem(*_):
            return "password"

        monkeypatch.setenv("ERT_STORAGE_TOKEN", "password")
        server_config_mock.__getitem__.side_effect = getitem
        return TestClient(app), subscribers

    return func


async def wait_for_server_to_complete(config):
    # Wait for the server to complete the optimization.
    # There should be a @pytest.mark.timeout(x) for tests that call this function.
    async def server_running():
        while True:
            event = await driver.event_queue.get()
            if isinstance(event, FinishedEvent) and event.iens == 0:
                return

    driver = await start_server(config, logging.DEBUG)
    wait_for_server(config.output_dir, 120)
    start_experiment(
        server_context=ServerConfig.get_server_context(config.output_dir),
        config=config,
    )
    await server_running()


def configure_everserver_logger(*args, **kwargs):
    """Mock exception raised"""
    raise Exception("Configuring logger failed")


@pytest.fixture
def mock_server(monkeypatch):
    def func(status: ExperimentState, message: str):
        server_patch = MagicMock()
        client_mock = MagicMock()
        response_mock = MagicMock()
        response_mock.json.return_value = {"status": status, "message": message}
        client_mock.get.return_value = response_mock
        server_patch.session.return_value.__enter__.return_value = client_mock
        monkeypatch.setattr("everest.detached.everserver.StorageService", server_patch)

    yield func


@pytest.mark.integration_test
def test_certificate_generation(change_to_tmpdir):
    cert, key, pw = everserver._generate_certificate(
        ServerConfig.get_certificate_dir("output")
    )

    # check that files are written
    assert os.path.exists(cert)
    assert os.path.exists(key)

    # check certificate is readable
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.load_cert_chain(cert, key, pw)  # raise on error


@pytest.mark.integration_test
@patch(
    "ert.shared.get_machine_name",
    return_value="A" * 67,
)
def test_certificate_generation_handles_long_machine_names(change_to_tmpdir):
    cert, key, pw = everserver._generate_certificate(
        ServerConfig.get_certificate_dir("output")
    )

    # check that files are written
    assert os.path.exists(cert)
    assert os.path.exists(key)

    # check certificate is readable
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.load_cert_chain(cert, key, pw)  # raise on error


@patch("sys.argv", ["name", "--output-dir", "everest_output"])
@patch(
    "everest.detached.everserver._configure_loggers",
    side_effect=configure_everserver_logger,
)
def test_configure_logger_failure(_, change_to_tmpdir):
    everserver.main()
    status = everserver_status(
        ServerConfig.get_everserver_status_path("everest_output")
    )

    assert status["status"] == ExperimentState.failed
    assert "Exception: Configuring logger failed" in status["message"]


@pytest.mark.integration_test
@patch("sys.argv", ["name", "--output-dir", "everest_output"])
@patch("everest.detached.everserver._configure_loggers")
def test_status_running_complete(_, change_to_tmpdir, mock_server):
    mock_server(ExperimentState.completed, "Optimization completed.")
    everserver.main()

    status = everserver_status(
        ServerConfig.get_everserver_status_path("everest_output")
    )

    assert status["status"] == ExperimentState.completed
    assert status["message"] == "Optimization completed."


@pytest.mark.integration_test
@patch("sys.argv", ["name", "--output-dir", "everest_output"])
@patch("everest.detached.everserver._configure_loggers")
def test_status_failed_job(_, change_to_tmpdir, mock_server):
    mock_server(ExperimentState.failed, OPT_FAILURE_REALIZATIONS)
    everserver.main()

    status = everserver_status(
        ServerConfig.get_everserver_status_path("everest_output")
    )

    # The server should fail and store a user-friendly message.
    assert status["status"] == ExperimentState.failed


@pytest.mark.integration_test
@pytest.mark.xdist_group(name="starts_everest")
@patch("sys.argv", ["name", "--output-dir", "everest_output"])
@patch("everest.detached.everserver._configure_loggers")
async def test_status_exception(_, change_to_tmpdir, min_config):
    min_config["simulator"] = {"queue_system": {"name": "local"}}
    config = EverestConfig(**min_config)

    await wait_for_server_to_complete(config)
    status = everserver_status(
        ServerConfig.get_everserver_status_path("everest_output")
    )

    assert status["status"] == ExperimentState.failed
    assert "Optimization failed:" in status["message"]


@pytest.mark.integration_test
@pytest.mark.xdist_group(name="starts_everest")
@pytest.mark.timeout(240)
@patch("sys.argv", ["name", "--output-dir", "everest_output"])
async def test_status_max_batch_num(copy_math_func_test_data_to_tmp):
    config = EverestConfig.load_file("config_minimal.yml")
    config_dict = {
        **config.model_dump(exclude_none=True),
        "optimization": {"algorithm": "optpp_q_newton", "max_batch_num": 1},
        "simulator": {"queue_system": {"name": "local", "max_running": 2}},
    }
    config = EverestConfig.model_validate(config_dict)

    await wait_for_server_to_complete(config)

    status = everserver_status(
        ServerConfig.get_everserver_status_path(config.output_dir)
    )

    # The server should complete without error.
    assert status["status"] == ExperimentState.completed
    assert status["message"] == "Maximum number of batches reached."
    storage = EverestStorage(Path(config.optimization_output_dir))
    storage.read_from_output_dir()

    # Check that there is only one batch.
    assert {b.batch_id for b in storage.data.batches} == {0}


@pytest.mark.integration_test
@pytest.mark.xdist_group(name="starts_everest")
@pytest.mark.timeout(240)
@patch("sys.argv", ["name", "--output-dir", "everest_output"])
async def test_status_too_few_realizations_succeeded(copy_math_func_test_data_to_tmp):
    config = EverestConfig.load_file("config_minimal.yml")
    config_dict = {
        **config.model_dump(exclude_none=True),
        "optimization": {"algorithm": "optpp_q_newton", "max_batch_num": 1},
        "simulator": {"queue_system": {"name": "local", "max_running": 2}},
        "model": {"realizations": [0, 1]},
    }
    config_dict["install_jobs"].append(
        {"name": "fail_simulation", "executable": "jobs/fail_simulation.py"}
    )
    config_dict["forward_model"].append("fail_simulation --fail geo_realization_0")
    config = EverestConfig.model_validate(config_dict)

    await wait_for_server_to_complete(config)

    status = everserver_status(
        ServerConfig.get_everserver_status_path(config.output_dir)
    )

    # The server should complete without error.
    assert status["status"] == ExperimentState.failed
    assert OPT_FAILURE_REALIZATIONS in status["message"]


@pytest.mark.integration_test
@pytest.mark.xdist_group(name="starts_everest")
@pytest.mark.timeout(240)
@patch("sys.argv", ["name", "--output-dir", "everest_output"])
async def test_status_all_realizations_failed(copy_math_func_test_data_to_tmp):
    config = EverestConfig.load_file("config_minimal.yml")
    config_dict = {
        **config.model_dump(exclude_none=True),
        "optimization": {"algorithm": "optpp_q_newton", "max_batch_num": 1},
        "simulator": {"queue_system": {"name": "local", "max_running": 2}},
    }
    config_dict["install_jobs"].append({"name": "fail", "executable": which("false")})
    config_dict["forward_model"].append("fail")
    config = EverestConfig.model_validate(config_dict)

    await wait_for_server_to_complete(config)

    status = everserver_status(
        ServerConfig.get_everserver_status_path(config.output_dir)
    )

    # The server should complete without error.
    assert status["status"] == ExperimentState.failed
    assert OPT_FAILURE_ALL_REALIZATIONS in status["message"]


@pytest.mark.integration_test
@pytest.mark.xdist_group(name="starts_everest")
@pytest.mark.timeout(240)
@patch("sys.argv", ["name", "--output-dir", "everest_output"])
async def test_status_contains_max_runtime_failure(change_to_tmpdir, min_config):
    min_config["simulator"] = {
        "queue_system": {"name": "local", "max_running": 2},
        "max_runtime": 1,
    }
    min_config["forward_model"] = ["sleep 5"]
    min_config["install_jobs"] = [{"name": "sleep", "executable": which("sleep")}]

    config = EverestConfig(**min_config)

    await wait_for_server_to_complete(config)

    status = everserver_status(
        ServerConfig.get_everserver_status_path("everest_output")
    )

    assert status["status"] == ExperimentState.failed
    assert "The run is cancelled due to reaching MAX_RUNTIME" in status["message"]


def test_websocket_no_authentication(monkeypatch, setup_client):
    client, _ = setup_client()
    with (
        client.websocket_connect("/experiment_server/events") as websocket,
        pytest.raises(WebSocketDisconnect) as exception,
    ):
        websocket.receive_json()
    assert exception.value.reason == "No authentication"


def test_websocket_wrong_password(monkeypatch, setup_client):
    client, _ = setup_client()
    credentials = b64encode(b"username:wrong_password").decode()
    with (
        client.websocket_connect(
            "/experiment_server/events",
            headers={"Authorization": f"Basic {credentials}"},
        ) as websocket,
        pytest.raises(WebSocketDisconnect) as exception,
    ):
        websocket.receive_json()
    assert not exception.value.reason


@pytest.mark.flaky(rerun=3)
def test_websocket_multiple_connections(monkeypatch, setup_client):
    client, subscribers = setup_client()
    credentials = b64encode(b"username:password").decode()
    with client.websocket_connect(
        "/experiment_server/events", headers={"Authorization": f"Basic {credentials}"}
    ) as websocket:
        event = websocket.receive_json()
        websocket.close()
    with client.websocket_connect(
        "/experiment_server/events", headers={"Authorization": f"Basic {credentials}"}
    ) as websocket:
        event_2 = websocket.receive_json()
    assert len(subscribers) == 2
    assert event == event_2


def test_websocket_multiple_connections_one_fails(monkeypatch, setup_client):
    client, subscribers = setup_client()
    credentials = b64encode(b"username:password").decode()
    with (
        client.websocket_connect("/experiment_server/events") as websocket,
        pytest.raises(WebSocketDisconnect),
    ):
        websocket.receive_json()
    with client.websocket_connect(
        "/experiment_server/events", headers={"Authorization": f"Basic {credentials}"}
    ) as websocket:
        event = websocket.receive_json()
    assert len(subscribers) == 1
    assert event == {"event_type": "EndEvent", "failed": False, "msg": "Complete"}


def test_websocket_multiple_events_in_queue(monkeypatch, setup_client):
    @dataclass
    class TestEvent:
        msg: str

    expected = [
        TestEvent("event_1"),
        TestEvent("event_2"),
        EndEvent(failed=False, msg="Done"),
    ]
    client, _ = setup_client(expected)
    credentials = b64encode(b"username:password").decode()
    event_msgs = []
    with client.websocket_connect(
        "/experiment_server/events", headers={"Authorization": f"Basic {credentials}"}
    ) as websocket:
        for _ in expected:
            event_msgs.append(websocket.receive_json())
    assert event_msgs == [jsonable_encoder(e) for e in expected]


async def test_websocket_no_events_on_connect(monkeypatch, setup_client):
    events = []
    client, subs = setup_client(events)
    credentials = b64encode(b"username:password").decode()
    result = []
    expected_result = EndEvent(failed=False, msg="Test message")

    with client.websocket_connect(
        "experiment_server/events", headers={"Authorization": f"Basic {credentials}"}
    ) as websocket:

        def receive_event():
            return websocket.receive_json()

        receive_task = asyncio.to_thread(receive_event)

        events.append(expected_result)
        for sub in subs.values():
            sub.notify()

        result.append(await receive_task)

    assert result == [jsonable_encoder(expected_result)]
