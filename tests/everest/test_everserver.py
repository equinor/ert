import asyncio
import json
import logging
import os
import ssl
from base64 import b64encode
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.encoders import jsonable_encoder
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import everest
from ert.ensemble_evaluator import EndEvent
from ert.run_models.everest_run_model import EverestExitCode
from ert.scheduler.event import FinishedEvent
from everest.config import EverestConfig, ServerConfig
from everest.detached import (
    ServerStatus,
    everserver_status,
    start_experiment,
    start_server,
    wait_for_server,
)
from everest.detached.jobs import everserver
from everest.detached.jobs.everserver import ExperimentComplete, _everserver_thread
from everest.everest_storage import EverestStorage


@pytest.fixture
def setup_client(monkeypatch):
    def func(events=None):
        events = [EndEvent(failed=False, msg="Complete")] if events is None else events
        uvicorn_mock = MagicMock()
        server_config_mock = MagicMock()

        def getitem(*_):
            return "password"

        server_config_mock.__getitem__.side_effect = getitem
        monkeypatch.setattr(everest.detached.jobs.everserver, "uvicorn", uvicorn_mock)
        subscribers = {}
        _everserver_thread(
            {
                "subscribers": subscribers,
                "events": events,
            },
            server_config_mock,
            MagicMock(),
        )
        # The first argument to uvicorn.run is app, so we extract that
        return TestClient(uvicorn_mock.run.mock_calls[0].args[0]), subscribers

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
    try:
        wait_for_server(config.output_dir, 120)
        start_experiment(
            server_context=ServerConfig.get_server_context(config.output_dir),
            config=config,
        )
    except (SystemExit, RuntimeError) as e:
        raise e
    await server_running()


def configure_everserver_logger(*args, **kwargs):
    """Mock exception raised"""
    raise Exception("Configuring logger failed")


@pytest.fixture
def mock_server(monkeypatch):
    def func(exit_code: EverestExitCode, message: str = ""):
        def server_mock(shared_data, server_config, msg_queue):
            msg_queue.put(ExperimentComplete(exit_code=exit_code, data=shared_data))
            _everserver_thread(shared_data, server_config, msg_queue)

        monkeypatch.setattr(
            everest.detached.jobs.everserver, "_everserver_thread", server_mock
        )

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


def test_hostfile_storage(change_to_tmpdir):
    host_file_path = "detach/.session/host_file"

    expected_result = {
        "host": "hostname.1.2.3",
        "port": "5000",
        "cert": "/a/b/c.cert",
        "auth": "1234",
    }
    everserver._write_hostfile(host_file_path, **expected_result)
    assert os.path.exists(host_file_path)
    with open(host_file_path, encoding="utf-8") as f:
        result = json.load(f)
    assert result == expected_result


@patch("sys.argv", ["name", "--output-dir", "everest_output"])
@patch(
    "everest.detached.jobs.everserver._configure_loggers",
    side_effect=configure_everserver_logger,
)
def test_configure_logger_failure(_, change_to_tmpdir):
    everserver.main()
    status = everserver_status(
        ServerConfig.get_everserver_status_path("everest_output")
    )

    assert status["status"] == ServerStatus.failed
    assert "Exception: Configuring logger failed" in status["message"]


@pytest.mark.integration_test
@patch("sys.argv", ["name", "--output-dir", "everest_output"])
@patch("everest.detached.jobs.everserver._configure_loggers")
def test_status_running_complete(_, change_to_tmpdir, mock_server):
    mock_server(EverestExitCode.COMPLETED)

    everserver.main()

    status = everserver_status(
        ServerConfig.get_everserver_status_path("everest_output")
    )

    assert status["status"] == ServerStatus.completed
    assert status["message"] == "Optimization completed."


@patch("sys.argv", ["name", "--output-dir", "everest_output"])
@patch("everest.detached.jobs.everserver._configure_loggers")
def test_status_failed_job(_, change_to_tmpdir, mock_server):
    mock_server(EverestExitCode.TOO_FEW_REALIZATIONS)
    everserver.main()

    status = everserver_status(
        ServerConfig.get_everserver_status_path("everest_output")
    )

    # The server should fail and store a user-friendly message.
    assert status["status"] == ServerStatus.failed


@pytest.mark.integration_test
@patch("sys.argv", ["name", "--output-dir", "everest_output"])
@patch("everest.detached.jobs.everserver._configure_loggers")
async def test_status_exception(_, change_to_tmpdir, min_config):
    config = EverestConfig(**min_config)

    await wait_for_server_to_complete(config)
    status = everserver_status(
        ServerConfig.get_everserver_status_path("everest_output")
    )

    assert status["status"] == ServerStatus.failed
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
    }
    config = EverestConfig.model_validate(config_dict)

    await wait_for_server_to_complete(config)

    status = everserver_status(
        ServerConfig.get_everserver_status_path(config.output_dir)
    )

    # The server should complete without error.
    assert status["status"] == ServerStatus.completed
    storage = EverestStorage(Path(config.optimization_output_dir))
    storage.read_from_output_dir()

    # Check that there is only one batch.
    assert {b.batch_id for b in storage.data.batches} == {0}


@pytest.mark.integration_test
@pytest.mark.xdist_group(name="starts_everest")
@pytest.mark.timeout(240)
@patch("sys.argv", ["name", "--output-dir", "everest_output"])
async def test_status_contains_max_runtime_failure(change_to_tmpdir, min_config):
    Path("SLEEP_job").write_text("EXECUTABLE sleep", encoding="utf-8")
    min_config["simulator"] = {"max_runtime": 1}
    min_config["forward_model"] = ["sleep 5"]
    min_config["install_jobs"] = [{"name": "sleep", "source": "SLEEP_job"}]

    config = EverestConfig(**min_config)

    await wait_for_server_to_complete(config)

    status = everserver_status(
        ServerConfig.get_everserver_status_path("everest_output")
    )

    assert status["status"] == ServerStatus.failed
    assert "The run is cancelled due to reaching MAX_RUNTIME" in status["message"]


def test_websocket_no_authentication(monkeypatch, setup_client):
    client, _ = setup_client()
    with (
        client.websocket_connect("/events") as websocket,
        pytest.raises(WebSocketDisconnect) as exception,
    ):
        websocket.receive_json()
    assert exception.value.reason == "No authentication"


def test_websocket_wrong_password(monkeypatch, setup_client):
    client, _ = setup_client()
    credentials = b64encode(b"username:wrong_password").decode()
    with (
        client.websocket_connect(
            "/events", headers={"Authorization": f"Basic {credentials}"}
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
        "/events", headers={"Authorization": f"Basic {credentials}"}
    ) as websocket:
        event = websocket.receive_json()
        websocket.close()
    with client.websocket_connect(
        "/events", headers={"Authorization": f"Basic {credentials}"}
    ) as websocket:
        event_2 = websocket.receive_json()
    assert len(subscribers) == 2
    assert event == event_2


def test_websocket_multiple_connections_one_fails(monkeypatch, setup_client):
    client, subscribers = setup_client()
    credentials = b64encode(b"username:password").decode()
    with (
        client.websocket_connect("/events") as websocket,
        pytest.raises(WebSocketDisconnect),
    ):
        websocket.receive_json()
    with client.websocket_connect(
        "/events", headers={"Authorization": f"Basic {credentials}"}
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
        "/events", headers={"Authorization": f"Basic {credentials}"}
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
        "/events", headers={"Authorization": f"Basic {credentials}"}
    ) as websocket:

        def receive_event():
            return websocket.receive_json()

        receive_task = asyncio.to_thread(receive_event)

        events.append(expected_result)
        for sub in subs.values():
            sub.notify()

        result.append(await receive_task)

    assert result == [jsonable_encoder(expected_result)]
