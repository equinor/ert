import glob
import json
import os
import socket
import ssl
from pathlib import Path
from unittest.mock import patch

import pytest

from ert.services import ErtServerController
from ert.services._storage_main import _create_connection_info, _generate_certificate
from ert.services.ert_server import create_ert_server_controller
from ert.shared import find_available_socket


@pytest.mark.skip_mac_ci  # Slow/failing - fqdn issue?
@pytest.mark.integration_test
def test_create_connection_string():
    authtoken = "very_secret_token"
    sock = find_available_socket()

    _create_connection_info(sock, authtoken, Path("path/to/cert"))

    assert "ERT_STORAGE_CONNECTION_STRING" in os.environ
    connection_string = json.loads(os.environ["ERT_STORAGE_CONNECTION_STRING"])
    assert "urls" in connection_string
    assert "authtoken" in connection_string
    assert (
        len(connection_string["urls"]) > 0
    )  # If we don't get a FQDN we may get only one url

    assert len(connection_string["urls"]) == len(
        set(connection_string["urls"])
    )  # all unique

    del os.environ["ERT_STORAGE_CONNECTION_STRING"]


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
                "127.0.0.1",
                socket.gethostname(),
                socket.getfqdn(),
            )
        ],
        "authtoken": "dummytoken",
    }

    with Path("storage_server.json").open(mode="w", encoding="utf-8") as f:
        json.dump(connection_info, f)
    create_ert_server_controller(project=Path(".").absolute())


@pytest.mark.skip_mac_ci  # Slow/failing - fqdn issue?
@patch("ert.services.ErtServerController.start_server")
def test_that_service_can_be_started_with_missing_cert_in_conn_info_json(
    start_server_mock, change_to_tmpdir
):
    """
    This is a regression test for a bug with the following reproduction steps:

        1. run `ert gui poly.ert` with an ert 14.3
        2. kill that process, meaning `storage_service_server.json` is not cleaned up.
        3. run `ert gui poly.ert` with ert 14.4.0, which
           looks for a 'cert' key present in storage_server.json
    """
    connection_info = {
        "urls": [
            f"http://{host}:51839"
            for host in (
                "127.0.0.1",
                socket.gethostname(),
                socket.getfqdn(),
            )
        ],
        "authtoken": "dummytoken",
    }
    with Path("storage_server.json").open(mode="w", encoding="utf-8") as f:
        json.dump(connection_info, f)
    ErtServerController.init_service(project=Path(".").absolute())
    start_server_mock.assert_called_once()


@patch("ert.services.ErtServerController.start_server")
def test_that_service_can_be_started_with_empty_conn_info_json(
    start_server_mock, change_to_tmpdir
):
    """An empty file on disk is an erroneous scenario in which we should
    ignore the file on disk and overwrite it by launching a new server"""
    Path("storage_server.json").touch()
    ErtServerController.init_service(project=Path(".").absolute())
    start_server_mock.assert_called_once()


@patch("ert.services.ErtServerController.start_server")
@patch("ert.services.ert_server.ErtServerController.fetch_url")
def test_that_stale_connection_info_file_is_removed_before_starting_new_service(
    fetch_url_mock, start_server_mock, change_to_tmpdir
):
    """Regression test: when a storage_server.json is left behind from a
    previous process (e.g. Ctrl+C killed the plotter without cleanup),
    init_service must delete the stale file before starting a new server.

    Without this, the new server's subprocess or any client calling
    create_ert_server_controller will read the stale file and try to connect
    to a dead server, causing a TimeoutError."""
    stale_file = Path("storage_server.json")
    connection_info = {
        "urls": ["http://127.0.0.1:1"],
        "authtoken": "stale_token",
        "cert": "",
        "host": "127.0.0.1",
        "port": "1",
        "auth": "",
    }
    stale_file.write_text(json.dumps(connection_info), encoding="utf-8")

    fetch_url_mock.side_effect = TimeoutError("server is dead")

    def assert_stale_file_deleted(**kwargs):
        assert not stale_file.exists(), (
            "Stale storage_server.json must be deleted before starting a new "
            "server, otherwise clients calling create_ert_server_controller "
            "will read dead connection info"
        )

    start_server_mock.side_effect = assert_stale_file_deleted

    ErtServerController.init_service(project=Path(".").absolute())
    start_server_mock.assert_called_once()


@patch("ert.services.ErtServerController.start_server")
def test_that_service_can_be_started_with_empty_json_content(
    start_server_mock, change_to_tmpdir
):
    """An empty JSON document on disk is an erroneous scenario in which we should
    ignore the file on disk and overwrite it by launching a new server"""
    Path("storage_server.json").write_text("{}", encoding="utf-8")
    ErtServerController.init_service(project=Path(".").absolute())
    start_server_mock.assert_called_once()


@pytest.mark.skip_mac_ci  # Slow/failing - fqdn issue?
@pytest.mark.integration_test
def test_storage_logging(change_to_tmpdir):
    """
    This is a regression test for a bug where the storage service
    would log everything twice
    """

    with ErtServerController.start_server(
        verbose=True,
        project=Path("."),
        parent_pid=os.getpid(),
    ) as server:
        assert server.wait_until_ready(), "StorageService failed to start"

    logfiles = glob.glob("api-log-storage*.txt")
    assert len(logfiles) == 1, "Expected exactly one log file"
    with Path(logfiles[0]).open(encoding="utf-8") as logfile:
        contents = logfile.readlines()

    # check for duplicated log entries
    assert (
        sum(
            "[INFO] ert.shared.storage.info: Starting dark storage" in e
            for e in contents
        )
        == 1
    ), "Found duplicated log entries"


@pytest.mark.skip_mac_ci  # Slow/failing - fqdn issue?
@pytest.mark.integration_test
def test_certificate_generation(change_to_tmpdir):
    cert, key, pw = _generate_certificate(Path())

    # check that files are written
    assert cert.exists()
    assert key.exists()

    # check certificate is readable
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.load_cert_chain(cert, key, pw)  # raise on error


@pytest.mark.skip_mac_ci  # Slow/failing - fqdn issue?
@pytest.mark.integration_test
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


@pytest.mark.skip_mac_ci  # Slow/failing - fqdn issue?
@pytest.mark.integration_test
def test_that_server_hosts_exists_as_san_in_certificate(change_to_tmpdir):
    auth_token = "very_secret_token"
    sock = find_available_socket()

    cert_path, _, _ = _generate_certificate(Path())

    conn_info = _create_connection_info(sock, auth_token, cert_path)
    # check certificate is readable
    x509 = ssl._ssl._test_decode_cert(conn_info["cert"])  # type: ignore[attr-defined]
    sans = [san[1] for san in x509["subjectAltName"]]

    # extract hostname from the url strings "https://<hostname>:<port>/..."
    hosts_from_urls = [u.split("https://")[1].split(":")[0] for u in conn_info["urls"]]

    assert set(sans) == set(hosts_from_urls)
    del os.environ["ERT_STORAGE_CONNECTION_STRING"]


def test_that_an_exception_is_raised_if_storage_server_file_has_no_permissions(
    change_to_tmpdir,
):
    file_path = Path("storage_server.json")
    file_path.write_text("{}", encoding="utf-8")
    mode = file_path.stat().st_mode
    os.chmod(file_path, 0o000)  # no permissions
    try:
        with pytest.raises(PermissionError):
            ErtServerController.init_service(project=Path(".").absolute())
    finally:
        os.chmod(file_path, mode)
