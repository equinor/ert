import glob
import json
import os
import socket

import pytest

from ert.services import StorageService
from ert.services._storage_main import _create_connection_info
from ert.shared import find_available_socket


def test_create_connection_string():
    authtoken = "very_secret_token"
    sock = find_available_socket()

    _create_connection_info(sock, authtoken, "path/to/cert")

    assert "ERT_STORAGE_CONNECTION_STRING" in os.environ
    connection_string = json.loads(os.environ["ERT_STORAGE_CONNECTION_STRING"])
    assert "urls" in connection_string
    assert "authtoken" in connection_string
    assert len(connection_string["urls"]) == 4

    del os.environ["ERT_STORAGE_CONNECTION_STRING"]


def test_that_service_can_be_started_with_existing_conn_info_json(tmp_path):
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

    with open(tmp_path / "storage_server.json", mode="w", encoding="utf-8") as f:
        json.dump(connection_info, f)
    StorageService.connect(project=tmp_path)


@pytest.mark.integration_test
def test_storage_logging(change_to_tmpdir):
    """
    This is a regression test for a bug where the storage service
    would log everything twice
    """

    with StorageService.start_server(
        verbose=True,
        project=".",
        parent_pid=os.getpid(),
    ) as server:
        server.wait_until_ready()

        logfiles = glob.glob("api-log-storage*.txt")
        assert len(logfiles) == 1, "Expected exactly one log file"
        with open(logfiles[0], encoding="utf-8") as logfile:
            contents = logfile.readlines()

        # check for duplicated log entries
        assert (
            sum(
                "[INFO] ert.shared.storage.info: Starting dark storage" in e
                for e in contents
            )
            == 1
        ), "Found duplicated log entries"
