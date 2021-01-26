import pytest
import signal
import requests
from textwrap import dedent
from pathlib import Path
from ert_shared.storage.server_monitor import (
    ServerBootFail,
    ServerMonitor,
    TimeoutError,
)
from ert_shared.storage import connection


@pytest.fixture
def server(request, monkeypatch, tmp_path: Path):
    marker = request.node.get_closest_marker("script")
    if marker is None:
        return

    script = (
        dedent(
            """\
    #!/usr/bin/python3
    import os
    import sys
    import time
    fd = os.environ.get("ERT_COMM_FD")
    if fd is not None: fd = int(fd)

    """
        )
        + marker.args[0]
    )
    path = tmp_path / "script"
    path.write_text(script)
    path.chmod(0o755)
    monkeypatch.setattr(ServerMonitor, "EXEC_ARGS", [str(path)])
    monkeypatch.setattr(ServerMonitor, "TIMEOUT", 5)

    proc = ServerMonitor()
    proc.start()
    yield proc


@pytest.mark.script("")
def test_init(server):
    server.join()


@pytest.mark.script("sys.exit(1)")
def test_fail(server):
    server.join()


@pytest.mark.script("")
def test_shutdown_after_finish(server):
    server.join()
    server.shutdown()


@pytest.mark.script(
    """\
os.write(fd, b'{"authtoken": "test123", "urls": ["url"]}')
"""
)
def test_info(server):
    assert server.fetch_connection_info() == {"authtoken": "test123", "urls": ["url"]}


@pytest.mark.script(
    """\
time.sleep(0.5)
os.write(fd, b'{"authtoken": "test123", "urls": ["url"]}')
"""
)
def test_info_slow(server):
    # fetch_connection_info() should block until this value is available
    assert server.fetch_connection_info() == {"authtoken": "test123", "urls": ["url"]}


@pytest.mark.script(
    """\
os.write(fd, b"This isn't valid json (I hope)")
"""
)
def test_authtoken_wrong_json(server):
    with pytest.raises(ServerBootFail):
        server.fetch_connection_info()


@pytest.mark.script(
    """\
os.write(fd, b'{"authtoken": "test123", "urls": ["url"]}')
os.close(fd)
time.sleep(10)
sys.exit(1)
"""
)
def test_long_lived(server):
    assert server.fetch_connection_info() == {"authtoken": "test123", "urls": ["url"]}
    assert server.shutdown() == -signal.SIGTERM


@pytest.mark.script(
    """\
time.sleep(10)
sys.exit(2)
"""
)
def test_not_respond(server):
    with pytest.raises(TimeoutError):
        server.fetch_connection_info()
    assert server.shutdown() == -signal.SIGTERM


@pytest.mark.script(
    """\
sys.exit(1)
"""
)
def test_authtoken_fail(server):
    with pytest.raises(Exception):
        server.fetch_connection_info()


def test_json_exists(tmpdir):
    with tmpdir.as_cwd():
        with open(str(tmpdir / "storage_server.json"), "w") as f:
            f.write("this is a json file")

        with pytest.raises(SystemExit):
            ServerMonitor()


def test_integration(request, tmpdir):
    """Actually start the server, wait for it to be online and do a health check"""
    with tmpdir.as_cwd():
        server = ServerMonitor()
        server.start()
        request.addfinalizer(lambda: server.shutdown())

        resp = requests.get(
            f"{server.fetch_url()}/healthcheck", auth=server.fetch_auth()
        )
        assert "date" in resp.json()

        # Use global connection info
        conn_info = connection.get_info()
        requests.get(conn_info["baseurl"] + "/healthcheck", auth=conn_info["auth"])

        # Use local connection info
        conn_info = connection.get_info(str(tmpdir))
        requests.get(conn_info["baseurl"] + "/healthcheck", auth=conn_info["auth"])

        server.shutdown()
        assert not (tmpdir / "storage_server.json").exists()

        # Global connection info no longer valid
        with pytest.raises(RuntimeError):
            connection.get_info()


def test_integration_auth(request, tmpdir):
    """Start the server, wait for it to be online and then do a health check with an
    invalid auth"""
    with tmpdir.as_cwd():
        server = ServerMonitor()
        server.start()
        request.addfinalizer(lambda: server.shutdown())

        # No auth
        resp = requests.get(f"{server.fetch_url()}/healthcheck")
        assert resp.status_code == 401

        # Invalid auth
        resp = requests.get(
            f"{server.fetch_url()}/healthcheck", auth=("__token__", "invalid-token")
        )
        assert resp.status_code == 403
