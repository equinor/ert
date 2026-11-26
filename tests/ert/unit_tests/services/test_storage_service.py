import json
import multiprocessing
import os
import socket
import ssl
import subprocess
import sys
from pathlib import Path
from unittest.mock import DEFAULT, patch

import pytest
from filelock import FileLock

from ert.services import ErtServerController
from ert.services._ert_server_main import _create_connection_info, _generate_certificate
from ert.services.ert_server_controller import (
    ServerAlreadyRunningError,
    create_ert_server_controller,
)
from ert.shared import find_available_socket
from ert.shared.constants import LOCALHOST_IP


@pytest.mark.slow
def test_create_connection_string(monkeypatch):
    authtoken = "very_secret_token"
    sock = find_available_socket()
    monkeypatch.setenv("ERT_SERVER_CONNECTION_STRING", "")

    _create_connection_info(sock, authtoken, Path("path/to/cert"))

    assert "ERT_SERVER_CONNECTION_STRING" in os.environ
    connection_string = json.loads(os.environ["ERT_SERVER_CONNECTION_STRING"])
    assert "urls" in connection_string
    assert "authtoken" in connection_string
    assert (
        len(connection_string["urls"]) > 0
    )  # If we don't get a FQDN we may get only one url

    assert len(connection_string["urls"]) == len(
        set(connection_string["urls"])
    )  # all unique


@pytest.mark.slow
def test_that_storage_log_config_keeps_loggers_created_before_it_is_applied(
    change_to_tmpdir, monkeypatch
):
    monkeypatch.delenv("ERT_LOG_DIR", raising=False)
    script = (
        "import logging, logging.config, yaml\n"
        "from ert.logging import STORAGE_LOG_CONFIG\n"
        "logger = logging.getLogger('ert.storage.local_storage')\n"
        "logging.config.dictConfig("
        "yaml.safe_load(STORAGE_LOG_CONFIG.read_text(encoding='utf-8')))\n"
        "print(logger.disabled)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"


def test_that_service_can_be_started_with_existing_conn_info_json(change_to_tmpdir):
    """
    This is a regression test for a bug with the following reproduction steps:

        1. run `ert gui snake_oil.ert` with an old version of ert
        2. kill that process, meaning `storage_service_server.json` is not cleaned up.
        3. run `ert gui snake_oil.ert` with latest version of ert, which
           may crash due to braking changes with respects to the file.
    """
    connection_info = {
        "urls": [
            f"http://{host}:51839"
            for host in (
                LOCALHOST_IP,
                socket.gethostname(),
                socket.getfqdn(),
            )
        ],
        "authtoken": "dummytoken",
    }

    with Path("ert_server_connection.json").open(mode="w", encoding="utf-8") as f:
        json.dump(connection_info, f)
    create_ert_server_controller(project=Path().absolute())


@patch("ert.services.ert_server_controller._Proc")
def test_that_stale_connection_info_file_is_removed_before_starting_new_service(
    server_process_mock, change_to_tmpdir
):
    """Regression test: when an ert_server_connection.json is left behind from a
    previous process (e.g. Ctrl+C killed the plotter without cleanup),
    init_service must delete the stale file before starting a new server.

    Without this, the new server's subprocess or any client calling
    create_ert_server_controller will read the stale file and try to connect
    to a dead server, causing a TimeoutError.
    """
    stale_file = Path("ert_server_connection.json")
    connection_info = {
        "urls": [f"http://{LOCALHOST_IP}:1"],
        "authtoken": "stale_token",
        "cert": "",
        "host": LOCALHOST_IP,
        "port": "1",
        "auth": "",
    }
    stale_file.write_text(json.dumps(connection_info), encoding="utf-8")

    def assert_stale_file_deleted(**kwargs):
        assert not stale_file.exists(), (
            "Stale ert_server_connection.json must be deleted before starting a new "
            "server, otherwise clients calling create_ert_server_controller "
            "will read dead connection info"
        )
        return DEFAULT

    server_process_mock.side_effect = assert_stale_file_deleted

    with ErtServerController.init_service(project=Path().absolute()):
        server_process_mock.assert_called_once()


@patch("ert.services.ert_server_controller._Proc")
def test_that_init_service_raises_and_keeps_connection_file_while_server_lock_is_held(
    server_process_mock, change_to_tmpdir
):
    connection_info_file = Path("ert_server_connection.json")
    connection_info_json = json.dumps(
        {
            "urls": [f"http://{LOCALHOST_IP}:51839"],
            "authtoken": "dummytoken",
            "cert": "",
            "host": LOCALHOST_IP,
            "port": "51839",
            "auth": "",
        }
    )
    connection_info_file.write_text(connection_info_json, encoding="utf-8")
    server_lock_of_other_process = FileLock("ert_server.lock")
    server_lock_of_other_process.acquire(timeout=0)

    try:
        with pytest.raises(ServerAlreadyRunningError, match="Server already running"):
            ErtServerController.init_service(project=Path().absolute())
    finally:
        server_lock_of_other_process.release()

    server_process_mock.assert_not_called()
    assert connection_info_file.read_text(encoding="utf-8") == connection_info_json


def _start_competing_server(project, start, stop, results):
    results.put(("ready", os.getpid()))
    if not start.wait(30):
        raise TimeoutError("Timed out waiting to start the server")
    try:
        with ErtServerController.start_server(
            project=project, parent_pid=os.getpid(), timeout=120
        ) as server:
            results.put(("started", server.fetch_connection_info()))
            if not stop.wait(120):
                raise TimeoutError("Timed out waiting to stop the server")
    except ServerAlreadyRunningError:
        results.put(("locked", None))


@pytest.mark.slow
def test_that_concurrent_controllers_publish_only_one_server_connection(
    tmp_path, change_to_tmpdir, monkeypatch
):
    monkeypatch.setenv("ERT_LOG_DIR", str(tmp_path))
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    stop = context.Event()
    results = context.Queue()
    processes = [
        context.Process(
            target=_start_competing_server, args=(tmp_path, start, stop, results)
        )
        for _ in range(2)
    ]
    started_processes = []
    try:
        for process in processes:
            process.start()
            started_processes.append(process)
        assert [results.get(timeout=30)[0] for _ in processes] == ["ready", "ready"]
        start.set()

        outcomes = [results.get(timeout=120) for _ in processes]
        assert sorted(outcome for outcome, _ in outcomes) == ["locked", "started"]
        connection_info = next(
            info for outcome, info in outcomes if outcome == "started"
        )
        connection_file = tmp_path / "ert_server_connection.json"
        assert (
            json.loads(connection_file.read_text(encoding="utf-8")) == connection_info
        )
    finally:
        stop.set()
        for process in started_processes:
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
        results.close()

    assert [process.exitcode for process in processes] == [0, 0]


@patch("ert.services.ert_server_controller._Proc")
def test_that_server_lock_can_be_reacquired_after_server_shutdown(
    server_process_mock, change_to_tmpdir
):
    with ErtServerController.init_service(project=Path().absolute()):
        pass

    server_lock = FileLock("ert_server.lock")
    server_lock.acquire(timeout=0)
    server_lock.release()

    with ErtServerController.init_service(project=Path().absolute()):
        assert server_process_mock.call_count == 2


@pytest.mark.slow
def test_storage_logging(change_to_tmpdir):
    """
    This is a regression test for a bug where the storage service
    would log everything twice
    """

    with ErtServerController.start_server(
        verbose=True,
        project=Path(),
        parent_pid=os.getpid(),
    ) as server:
        assert server.wait_until_ready(), "StorageService failed to start"

    logfiles = list(Path().glob("api-log-storage*.txt"))
    assert len(logfiles) == 1, "Expected exactly one log file"
    contents = logfiles[0].read_text(encoding="utf-8").splitlines()

    # check for duplicated log entries
    assert (
        sum(
            "[INFO] ert.shared.storage.info: Starting ert server" in e for e in contents
        )
        == 1
    ), "Found duplicated log entries"


@pytest.mark.slow
def test_certificate_generation(change_to_tmpdir):
    cert, key, pw = _generate_certificate(Path())

    # check that files are written
    assert cert.exists()
    assert key.exists()

    # check certificate is readable
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.load_cert_chain(cert, key, pw)  # raise on error


@pytest.mark.slow
def test_certificate_generation_handles_long_machine_names(change_to_tmpdir):
    with patch(
        "ert.shared.get_machine_name",
        return_value="A" * 67,
    ):
        cert, key, pw = _generate_certificate(Path())

    # check that files are written
    assert cert.exists()
    assert key.exists()

    # check certificate is readable
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.load_cert_chain(cert, key, pw)  # raise on error


@pytest.mark.slow
def test_that_server_hosts_and_localhost_ip_exists_as_san_in_certificate(
    change_to_tmpdir, monkeypatch
):
    auth_token = "very_secret_token"
    sock = find_available_socket()
    cert_path, _, _ = _generate_certificate(Path())

    conn_info = _create_connection_info(sock, auth_token, cert_path)
    # check certificate is readable
    x509 = ssl._ssl._test_decode_cert(conn_info["cert"])  # type: ignore[attr-defined]
    dns_sans = [value for kind, value in x509["subjectAltName"] if kind == "DNS"]

    # extract hostname from the url strings "https://<hostname>:<port>/..."
    hosts_from_urls = [u.split("https://")[1].split(":")[0] for u in conn_info["urls"]]
    assert set(dns_sans) == set(hosts_from_urls)
    assert ("IP Address", LOCALHOST_IP) in x509["subjectAltName"]


def test_that_an_exception_is_raised_if_storage_server_file_has_no_permissions(
    change_to_tmpdir,
):
    file_path = Path("ert_server.lock")
    file_path.touch()
    mode = file_path.stat().st_mode
    Path(file_path).chmod(0o000)  # no permissions
    try:
        with pytest.raises(PermissionError):
            ErtServerController.init_service(project=Path().absolute())
    finally:
        Path(file_path).chmod(mode)
