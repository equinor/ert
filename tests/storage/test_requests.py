import os
import pytest

from pathlib import Path
from ert_shared.storage import requests
from ert_shared.storage.server_monitor import ServerMonitor


@pytest.fixture(scope="module")
def server(request, db_populated):
    cwd = os.getcwd()
    path = db_populated.path
    os.chdir(path)

    server_ = ServerMonitor()
    server_.start()
    request.addfinalizer(lambda: server_.shutdown())

    server_.fetch_connection_info()
    yield server_

    os.chdir(cwd)


def test_connect(server):
    with pytest.raises(ValueError) as exc:
        requests.get("healthcheck")
    assert "wasn't initialized" in str(exc.value)

    requests.connect(Path.cwd())
    resp = requests.get("healthcheck")
    assert resp.status_code == 200
