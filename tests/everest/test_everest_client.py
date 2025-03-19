import logging
import ssl
import threading
from pathlib import Path

import pytest
import requests
import uvicorn
from fastapi import FastAPI
from starlette.responses import Response

from everest.bin.everest_script import everest_entry
from everest.config import EverestConfig, ServerConfig
from everest.detached import server_is_running
from everest.detached.jobs.everserver import _find_open_port
from everest.gui.everest_client import EverestClient
from everest.strings import STOP_ENDPOINT
from tests.ert.utils import wait_until


@pytest.fixture
def client_server_mock() -> tuple[FastAPI, threading.Thread, EverestClient]:
    server_app = FastAPI()
    host = "127.0.0.1"
    port = _find_open_port(host, lower=5000, upper=5800)
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

                return True
            except requests.exceptions.ConnectionError:
                return False

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

    @server_app.post(f"/{STOP_ENDPOINT}")
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

    @server_app.post(f"/{STOP_ENDPOINT}")
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


@pytest.mark.integration_test
@pytest.mark.xdist_group("math_func/config_minimal.yml")
def test_that_multiple_everest_clients_can_connect_to_server(cached_example):
    # We use a cached run for the reference list of received events
    path, config_file, _, server_events_list = cached_example(
        "math_func/config_minimal.yml"
    )

    config_path = Path(path) / config_file
    ever_config = EverestConfig.load_file(config_path)

    # Run the case through everserver
    everest_main_thread = threading.Thread(
        target=everest_entry, args=[[str(config_path), "--skip-prompt"]]
    )

    everest_main_thread.start()

    def everserver_is_running():
        return server_is_running(
            *ServerConfig.get_server_context(ever_config.output_dir)
        )

    wait_until(everserver_is_running, interval=1, timeout=20)

    server_context = ServerConfig.get_server_context(ever_config.output_dir)
    url, cert, auth = server_context

    ssl_context = ssl.create_default_context()
    ssl_context.load_verify_locations(cafile=cert)
    username, password = auth

    client_event_queues = []
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
        monitor_thread.start()

    # Wait until the server has finished running the simulation
    everest_main_thread.join()

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
