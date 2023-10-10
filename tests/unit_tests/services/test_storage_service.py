import json
import os

import pytest
import requests
from flaky import flaky

from ert.services import StorageService, _storage_main
from ert.shared import port_handler


@flaky
@pytest.mark.integration_test
@pytest.mark.requires_ert_storage
def test_integration(tmp_path, monkeypatch):
    """Actually start the server, wait for it to be online and do a health check"""
    monkeypatch.chdir(tmp_path)

    # Note: Sqlite needs at least 4-5 seconds to spin up even on
    # an unloaded M1-based Mac using local disk. On the CI-server
    # we have less control of available resources, so set timeout-
    # value large to allow time for sqlite to get ready
    with StorageService.start_server(timeout=120) as server:
        resp = requests.get(
            f"{server.fetch_url()}/healthcheck", auth=server.fetch_auth()
        )
        assert "ALL OK!" in resp.json()

        with StorageService.session() as session:
            session.get("/healthcheck")

    assert not (tmp_path / "storage_server.json").exists()


@pytest.mark.requires_ert_storage
def test_integration_timeout(tmp_path, monkeypatch):
    """Try to start the server but give it too small time to get ready and
    expect a timeout"""
    monkeypatch.chdir(tmp_path)

    with pytest.raises(TimeoutError), StorageService.start_server(
        timeout=0.01
    ) as server:
        # Note timeout-value here in context of note above
        requests.get(f"{server.fetch_url()}/healthcheck", auth=server.fetch_auth())

    assert not (tmp_path / "storage_server.json").exists()


def test_create_connection_string(monkeypatch):
    authtoken = "very_secret_token"
    _, _, sock = port_handler.find_available_port()

    _storage_main._create_connection_info(sock, authtoken)

    assert "ERT_STORAGE_CONNECTION_STRING" in os.environ
    connection_string = json.loads(os.environ["ERT_STORAGE_CONNECTION_STRING"])
    assert "urls" in connection_string
    assert "authtoken" in connection_string
    assert len(connection_string["urls"]) == 3

    del os.environ["ERT_STORAGE_CONNECTION_STRING"]


@pytest.mark.requires_ert_storage
def test_init_service_no_server_start(tmpdir, mock_start_server, mock_connect):
    with tmpdir.as_cwd():
        StorageService.init_service(project=str(tmpdir))
        mock_connect.assert_called_once_with(project=str(tmpdir), timeout=0)
        mock_start_server.assert_not_called()


@pytest.mark.requires_ert_storage
def test_init_service_server_start_if_no_conf_file(tmpdir, mock_start_server):
    with tmpdir.as_cwd():
        StorageService.init_service(project=str(tmpdir))
        mock_start_server.assert_called_once_with(project=str(tmpdir))


@pytest.mark.requires_ert_storage
def test_init_service_server_start_conf_info_is_stale(
    tmp_path, mock_start_server, monkeypatch
):
    (tmp_path / f"{StorageService.service_name}_server.json").write_text(
        json.dumps({"authtoken": "test123", "urls": ["http://127.0.0.1:51821"]}),
        encoding="utf-8",
    )

    def raise_conerr(x, auth):
        raise requests.ConnectionError()

    monkeypatch.setattr(requests, "get", raise_conerr)

    with tmp_path:
        StorageService.init_service(project=tmp_path)
        mock_start_server.assert_called_once_with(project=tmp_path)
