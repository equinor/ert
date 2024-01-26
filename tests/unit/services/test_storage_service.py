import json
import os

from ert.services import _storage_main
from ert.shared import port_handler


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
