import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from textwrap import dedent

import pytest

from ert.services._base_service import (
    SERVICE_CONF_PATHS,
    BaseService,
    ServerBootFail,
    cleanup_service_files,
    local_exec_args,
)


class _DummyService(BaseService):
    service_name = "dummy"

    def __init__(self, exec_args, *args, **kwargs):
        super().__init__(exec_args=exec_args, timeout=10, *args, **kwargs)

    def start(self):
        """Helper function for non-singleton testing"""
        assert self._proc is not None
        self._proc.start()

    def join(self):
        """Helper function for non-singleton testing"""
        self.wait()


@pytest.fixture
def server_script(monkeypatch, tmp_path: Path, request):
    marker = request.node.get_closest_marker("script")
    if marker is None:
        return None

    monkeypatch.chdir(tmp_path)

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
    return path


@pytest.fixture
def server(monkeypatch, server_script, request):
    proc = _DummyService([str(server_script)])
    request.addfinalizer(proc.shutdown)
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
time.sleep(0.5)
os.write(fd, b'{"authtoken": "test123", "urls": ["url"]}')
"""
)
def test_info_slow(server):
    # fetch_conn_info() should block until this value is available
    assert server.fetch_conn_info() == {"authtoken": "test123", "urls": ["url"]}


@pytest.mark.script(
    """\
os.write(fd, b"This isn't valid json (I hope)")
"""
)
def test_authtoken_wrong_json(server):
    with pytest.raises(ServerBootFail):
        server.fetch_conn_info()


@pytest.mark.script(
    """\
os.write(fd, b'{"authtoken": "test123", "urls": ["url"]}')
os.close(fd)
time.sleep(10)
sys.exit(1)
"""
)
def test_long_lived(server, tmp_path):
    assert server.fetch_conn_info() == {"authtoken": "test123", "urls": ["url"]}
    assert server.shutdown() == -signal.SIGTERM
    assert not (tmp_path / "dummy_server.json").exists()


@pytest.mark.script(
    """\
time.sleep(30)
sys.exit(2)
"""
)
def test_not_respond(server, caplog):
    server._timeout = 1
    with pytest.raises(TimeoutError):
        with caplog.at_level(logging.CRITICAL):
            server.fetch_conn_info()
            assert "startup exceeded defined timeout" in caplog.text
    assert server.shutdown() == -signal.SIGTERM


@pytest.mark.script(
    """\
sys.exit(1)
"""
)
def test_authtoken_fail(server):
    with pytest.raises(Exception):
        server.fetch_conn_info()


@pytest.mark.script(
    """\
os.write(fd, b'{"authtoken": "test123", "urls": ["url"]}')
os.close(fd)
time.sleep(10)  # Wait for the test to read the JSON file
"""
)
def test_json_created(server):
    server.fetch_conn_info()  # wait for it to start

    with open("dummy_server.json", encoding="utf-8") as f:
        assert f.read()


@pytest.mark.script(
    """\
os.write(fd, b'{"authtoken": "test123", "urls": ["url"]}')
os.close(fd)
"""
)
def test_json_deleted(server):
    """
    _BaseService is responsible for deleting the JSON file after the
    subprocess is finished running.
    """
    server.fetch_conn_info()  # wait for it to start
    time.sleep(2)  # ensure subprocess is done before calling shutdown()

    assert not os.path.exists("dummy_server.json")


@pytest.mark.script(
    """\
os.write(fd, b'{"authtoken": "test123", "urls": ["url"]}')
os.close(fd)
time.sleep(10) # ensure "server" doesn't exit before test
"""
)
def test_singleton_start(server_script, tmp_path):
    with _DummyService.start_server(exec_args=[str(server_script)]) as service:
        assert service.wait_until_ready()
        assert (tmp_path / "dummy_server.json").exists()

    assert not (tmp_path / "dummy_server.json").exists()


@pytest.mark.script(
    """\
time.sleep(1)
os.write(fd, b'{"authtoken": "test123", "urls": ["url"]}')
os.close(fd)
"""
)
def test_singleton_connect(server_script):
    with _DummyService.start_server(exec_args=[str(server_script)]) as server:
        client = _DummyService.connect(timeout=30)
        assert server is client


@pytest.mark.script(
    """\
os.write(fd, b'{"authtoken": "test123", "urls": ["url"]}')
os.close(fd)
time.sleep(10) # ensure "server" doesn't exit before test
"""
)
def test_singleton_connect_early(server_script, tmp_path):
    """
    Tests that a connection can be attempted even if it's started _before_
    the server exists
    """
    start_event = threading.Event()
    ready_event = threading.Event()

    # .connect() will block while it tries to connect
    # Do it in a separate thread
    class ClientThread(threading.Thread):
        def run(self):
            start_event.set()
            try:
                self.client = _DummyService.connect(timeout=30)
            except Exception as ex:
                self.exception = ex
            ready_event.set()

    client_thread = ClientThread()
    client_thread.start()

    start_event.wait()  # Client thread has started
    with _DummyService.start_server(exec_args=[str(server_script)]) as server:
        ready_event.wait()  # Client thread has connected to server
        assert not getattr(
            client_thread, "exception", None
        ), f"Exception from connect: {client_thread.exception}"
        client = client_thread.client
        assert client is not server
        assert client.fetch_conn_info() == server.fetch_conn_info()

    assert not (tmp_path / "dummy_server.json").exists()


@pytest.mark.script(
    """\
os.write(fd, b'{"authtoken": "test123", "urls": ["url"]}')
os.close(fd)
"""
)
def test_singleton_start_or_connect(server_script):
    """
    Tests that a connection can be attempted even if it's started _before_
    the server exists
    """
    with _DummyService.connect_or_start_server(
        exec_args=[str(server_script)]
    ) as server:
        assert server.fetch_conn_info() == {"authtoken": "test123", "urls": ["url"]}


@pytest.mark.script(
    """\
os.write(fd, b'{"authtoken": "test123", "urls": ["url"]}')
os.close(fd)
time.sleep(10) # ensure "server" doesn't exit before test
"""
)
def test_singleton_start_or_connect_exists(server_script):
    """
    Tests that a connection can be attempted even if it's started _before_
    the server exists
    """
    with _DummyService.start_server([str(server_script)]) as server:
        server.wait_until_ready()
        with _DummyService.connect_or_start_server([str(server_script)]) as client:
            assert server.fetch_conn_info() == client.fetch_conn_info()


@pytest.mark.parametrize("script,should_exist", [("storage", True), ("foobar", False)])
def test_local_exec_args(script, should_exist):
    exec_args = local_exec_args(script)
    assert len(exec_args) == 2
    assert exec_args[0] == sys.executable
    assert Path(exec_args[1]).is_file() == should_exist


def test_local_exec_args_multi():
    exec_args = local_exec_args(["storage", "foo", "-bar"])
    assert len(exec_args) == 4
    assert exec_args[0] == sys.executable
    assert exec_args[2] == "foo"
    assert exec_args[3] == "-bar"


def test_cleanup_service_files(tmpdir):
    with tmpdir.as_cwd():
        storage_service_name = "storage"
        storage_service_file = f"{storage_service_name}_server.json"
        with open(storage_service_file, "w", encoding="utf-8") as f:
            f.write("storage_service info")
        assert Path(storage_service_file).exists()
        SERVICE_CONF_PATHS.add(tmpdir / storage_service_file)

        webviz_service_name = "webviz-ert"
        webviz_service_file = f"{webviz_service_name}_server.json"
        with open(webviz_service_file, "w", encoding="utf-8") as f:
            f.write("webviz-ert info")
        assert Path(webviz_service_file).exists()
        SERVICE_CONF_PATHS.add(tmpdir / webviz_service_file)

        with pytest.raises(OSError):
            cleanup_service_files(signum=99, frame=None)

        assert not Path(storage_service_file).exists()
        assert not Path(webviz_service_file).exists()
