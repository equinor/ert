import logging
import ssl
import sys
import threading
import warnings
from pathlib import Path

import pytest
import requests
import uvicorn
import yaml
from fastapi import FastAPI
from starlette.responses import Response

from ert.services import ErtServerConnection
from ert.shared import find_available_socket
from everest.bin.everest_script import everest_entry
from everest.config import EverestConfig, ServerConfig
from everest.detached import server_is_running
from everest.gui.everest_client import EverestClient
from everest.strings import EverEndpoints
from tests.ert.utils import wait_until


@pytest.fixture
def client_server_mock() -> tuple[FastAPI, threading.Thread, EverestClient]:
    server_app = FastAPI()
    host = "127.0.0.1"
    port = find_available_socket(host, range(5000, 5800)).getsockname()[1]
    server_url = f"http://{host}:{port}"

    @server_app.get("alive")
    def alive():
        return Response("Hello", status_code=200)

    server = uvicorn.Server(
        uvicorn.Config(server_app, host=host, port=port, log_level="info")
    )

    server_thread = threading.Thread(
        target=server.run,
        daemon=True,
    )

    everest_client = EverestClient(
        url=server_url,
        cert_file="N/A",
        username="",
        password="",
        ssl_context=ssl.create_default_context(),
    )

    def wait_until_alive(timeout=60, sleep_between_retries=1) -> None:
        def ping_server() -> bool:
            try:
                requests.get(
                    f"{server_url}/alive",
                    verify="N/A",
                    auth=("", ""),
                    proxies={"http": None, "https": None},  # type: ignore
                )
            except requests.exceptions.ConnectionError:
                return False
            else:
                return True

        # These warnings emitted by uvicorn, which is still using legacy
        # websockets. This is a known issue, and does not cause problems in the
        # main code. (see: https://github.com/encode/uvicorn/discussions/2476)
        # Hence we ignore them in the tests. Potentially, this may be
        # removed when this is resolved within uvicorn.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="websockets.legacy is deprecated")
            warnings.filterwarnings(
                "ignore",
                message="websockets.server.WebSocketServerProtocol is deprecated",
            )
            wait_until(ping_server, timeout=timeout, interval=sleep_between_retries)

    yield server_app, server_thread, everest_client, wait_until_alive

    if server_thread.is_alive():
        server.should_exit = True
        server_thread.join()


@pytest.mark.integration_test
@pytest.mark.flaky(rerun=2)
def test_that_stop_invokes_correct_endpoint(
    caplog, client_server_mock: tuple[FastAPI, threading.Thread, EverestClient]
):
    server_app, server_thread, client, wait_until_alive = client_server_mock

    @server_app.post(f"/{EverEndpoints.stop}")
    def stop():
        return Response("STOP..", 200)

    server_thread.start()
    wait_until_alive()

    with caplog.at_level(logging.INFO):
        client.stop()

    assert "Cancelled experiment from Everest" in caplog.messages
    server_thread.should_exit = True


@pytest.mark.integration_test
def test_that_stop_errors_on_non_ok_httpcode(
    caplog, client_server_mock: tuple[FastAPI, threading.Thread, EverestClient]
):
    server_app, server_thread, client, wait_until_alive = client_server_mock

    @server_app.post(f"/{EverEndpoints.stop}")
    def stop():
        return Response("STOP..", 505)

    server_thread.start()
    wait_until_alive()

    with caplog.at_level(logging.ERROR):
        client.stop()

    assert any(
        "Failed to cancel Everest experiment" in m
        and "server responded with status 505" in m
        for m in caplog.messages
    )


def test_that_stop_errors_on_server_down(
    caplog, client_server_mock: tuple[FastAPI, threading.Thread, EverestClient]
):
    _, _, client, _ = client_server_mock

    with caplog.at_level(logging.ERROR):
        client.stop()

    assert any(
        "Connection error when cancelling Everest experiment" in m
        for m in caplog.messages
    )


@pytest.mark.integration_test
def test_that_stop_errors_on_server_up_but_endpoint_down(
    caplog, client_server_mock: tuple[FastAPI, threading.Thread, EverestClient]
):
    _, server_thread, client, wait_until_alive = client_server_mock

    server_thread.start()
    wait_until_alive()

    with caplog.at_level(logging.ERROR):
        client.stop()

    assert any(
        "server responded with status 404: Not Found" in m for m in caplog.messages
    )


@pytest.mark.skip_mac_ci
@pytest.mark.integration_test
@pytest.mark.xdist_group("math_func/config_minimal.yml")
@pytest.mark.flaky(rerun=3)
@pytest.mark.skipif(
    sys.version_info[0:3] == (3, 13, 6), reason="Fails on Python 3.13.6"
)
def test_that_multiple_everest_clients_can_connect_to_server(
    cached_example, change_to_tmpdir
):
    # We use a cached run for the reference list of received events
    path, config_file, _, server_events_list = cached_example(
        "math_func/config_minimal.yml"
    )

    config_path = Path(path) / config_file
    config_content = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_content["simulator"] = {"queue_system": {"name": "local", "max_running": 2}}
    config_path.write_text(
        yaml.dump(config_content, default_flow_style=False), encoding="utf-8"
    )

    ever_config = EverestConfig.load_file(config_path)

    # Run the case through everserver
    everest_main_thread = threading.Thread(
        target=everest_entry, args=[[str(config_path), "--skip-prompt"]]
    )

    everest_main_thread.start()
    client = ErtServerConnection.session(
        Path(ServerConfig.get_session_dir(ever_config.output_dir))
    )

    def everserver_is_running():
        return server_is_running(
            *ServerConfig.get_server_context_from_conn_info(client.conn_info)
        )

    wait_until(everserver_is_running, interval=1, timeout=300)

    server_context = ServerConfig.get_server_context_from_conn_info(client.conn_info)
    url, cert, auth = server_context

    ssl_context = ssl.create_default_context()
    ssl_context.load_verify_locations(cafile=cert)
    username, password = auth

    client_event_queues = []
    monitor_threads = []
    for _ in range(5):
        client = EverestClient(
            url=url,
            cert_file=cert,
            username=username,
            password=password,
            ssl_context=ssl_context,
        )

        # Connect to the websockets endpoint
        client_event_queue, monitor_thread = client.setup_event_queue_from_ws_endpoint()
        client_event_queues.append(client_event_queue)
        monitor_threads.append(monitor_thread)
        monitor_thread.start()

    # Wait until the server has finished running the simulation
    everest_main_thread.join()
    for _thread in monitor_threads:
        if _thread.is_alive():
            _thread.join(timeout=5)

    # Expect all the clients to hold the same events
    client_event_lists = []
    for event_queue in client_event_queues:
        event_list = []
        while not event_queue.empty():
            event_list.append(event_queue.get())

        client_event_lists.append(event_list)

    first = client_event_lists[0]
    assert all(first == other for other in client_event_lists[1:])

    first_everevents = [e.event_type for e in first if "Everest" in e.event_type]
    server_everevents = [
        e.event_type for e in server_events_list if "Everest" in e.event_type
    ]

    # Compare only everest events, as the events from the forward model
    # are (at time of writing) not deterministic enough to expect equality
    assert first_everevents == server_everevents
