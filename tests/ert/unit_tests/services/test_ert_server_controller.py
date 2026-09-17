import signal
import threading
import time
from pathlib import Path
from textwrap import dedent

import pytest

from ert.services import ert_server_controller
from ert.services.ert_server_controller import (
    _ERT_SERVER_CONNECTION_INFO_FILE,
    SERVICE_CONF_PATHS,
    ErtServerController,
    ServerBootFail,
    cleanup_service_files,
    create_ert_server_controller,
)


class _DummyService(ErtServerController):
    service_name = "dummy"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **({"storage_path": ".", "timeout": 10} | kwargs))

    def start(self):
        """Helper function for non-singleton testing"""
        assert self._thread_that_starts_server_process is not None
        self._thread_that_starts_server_process.start()

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
    #!/usr/bin/env python3
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
def server(monkeypatch, server_script):
    monkeypatch.setattr(
        ert_server_controller, "_ERT_SERVER_EXECUTABLE_FILE", server_script
    )
    proc = _DummyService()
    proc.start()
    yield proc
    proc.shutdown()


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


@pytest.mark.slow
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
def test_long_lived(server, tmp_path):
    assert server.fetch_connection_info() == {"authtoken": "test123", "urls": ["url"]}
    assert server.shutdown() == -signal.SIGTERM
    assert not (tmp_path / _ERT_SERVER_CONNECTION_INFO_FILE).exists()


@pytest.mark.slow
@pytest.mark.script(
    """\
time.sleep(10)
sys.exit(2)
"""
)
def test_that_fetch_connection_info_times_out_when_server_does_not_respond(
    monkeypatch, server_script
):
    monkeypatch.setattr(
        ert_server_controller, "_ERT_SERVER_EXECUTABLE_FILE", server_script
    )
    server = _DummyService(timeout=2)
    server.start()
    try:
        with pytest.raises(TimeoutError):
            server.fetch_connection_info()
    finally:
        returncode = server.shutdown()
    assert returncode == -signal.SIGTERM


@pytest.mark.script(
    """\
sys.exit(1)
"""
)
def test_authtoken_fail(server):
    with pytest.raises(ServerBootFail):
        server.fetch_connection_info()


@pytest.mark.script(
    """\
os.write(fd, b'{"authtoken": "test123", "urls": ["url"]}')
os.close(fd)
time.sleep(10)  # Wait for the test to read the JSON file
"""
)
def test_json_created(server):
    server.fetch_connection_info()  # wait for it to start

    assert Path(_ERT_SERVER_CONNECTION_INFO_FILE).read_text(encoding="utf-8")


@pytest.mark.slow
@pytest.mark.script(
    """\
os.write(fd, b'{"authtoken": "test123", "urls": ["url"]}')
os.close(fd)
"""
)
def test_json_deleted(server):
    """
    ErtServer is responsible for deleting the JSON file after the
    subprocess is finished running.
    """
    server.fetch_connection_info()  # wait for it to start
    time.sleep(2)  # ensure subprocess is done before calling shutdown()

    assert not Path(_ERT_SERVER_CONNECTION_INFO_FILE).exists()


@pytest.mark.script(
    """\
os.write(fd, b'{"authtoken": "test123", "urls": ["url"]}')
os.close(fd)
time.sleep(10) # ensure "server" doesn't exit before test
"""
)
def test_singleton_start(monkeypatch, server_script, tmp_path):
    monkeypatch.setattr(
        ert_server_controller, "_ERT_SERVER_EXECUTABLE_FILE", server_script
    )
    with _DummyService.start_server(".", timeout=10) as service:
        assert service.wait_until_ready()
        assert (tmp_path / _ERT_SERVER_CONNECTION_INFO_FILE).exists()

    assert not (tmp_path / _ERT_SERVER_CONNECTION_INFO_FILE).exists()


@pytest.mark.script(
    """\
time.sleep(1)
os.write(fd, b'{"authtoken": "test123", "urls": ["url"]}')
os.close(fd)
"""
)
def test_that_connect_logs_permission_error(tmp_path, caplog):
    tmp_path.chmod(0o000)
    caplog.clear()
    caplog.set_level("ERROR")
    with pytest.raises(PermissionError):
        create_ert_server_controller(project=tmp_path, timeout=30)

    tmp_path.chmod(0o755)

    assert len(caplog.records) == 1
    assert (
        "cannot connect to ert server service due to permission issues." in caplog.text
    )


@pytest.mark.slow
@pytest.mark.script(
    """\
os.write(fd, b'{"authtoken": "test123", "urls": ["url"]}')
os.close(fd)
time.sleep(10) # ensure "server" doesn't exit before test
"""
)
def test_singleton_connect_early(server_script, tmp_path, monkeypatch):
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
                self.controller = create_ert_server_controller(
                    project=tmp_path, timeout=30
                )
            except Exception as ex:
                self.exception = ex
            ready_event.set()

    client_thread = ClientThread()
    client_thread.start()

    start_event.wait()  # Client thread has started
    monkeypatch.setattr(
        ert_server_controller, "_ERT_SERVER_EXECUTABLE_FILE", server_script
    )
    with _DummyService.start_server(".") as server:
        ready_event.wait()  # Client thread has connected to server
        assert not getattr(client_thread, "exception", None), (
            f"Exception from connect: {client_thread.exception}"
        )
        client = client_thread.controller
        assert client is not server
        assert client.fetch_connection_info() == server.fetch_connection_info()

    assert not (tmp_path / _ERT_SERVER_CONNECTION_INFO_FILE).exists()


@pytest.mark.script(
    """\
os.write(fd, b"This isn't valid json (I hope)")
"""
)
def test_that_wait_until_ready_returns_false_on_boot_failure(server):
    assert server.wait_until_ready(timeout=10) is False


@pytest.mark.slow
@pytest.mark.script(
    """\
time.sleep(10)
sys.exit(2)
"""
)
def test_that_wait_until_ready_returns_false_on_timeout(monkeypatch, server_script):
    monkeypatch.setattr(
        ert_server_controller, "_ERT_SERVER_EXECUTABLE_FILE", server_script
    )
    server = _DummyService(timeout=2)
    server.start()
    try:
        assert server.wait_until_ready(timeout=1) is False
    finally:
        server.shutdown()


@pytest.mark.script(
    """\
os.write(fd, b'{"authtoken": "test123", "urls": ["url"]}')
os.close(fd)
time.sleep(10)
"""
)
def test_that_fetch_connection_info_raises_when_storage_path_does_not_exist(
    monkeypatch, server_script, tmp_path
):
    monkeypatch.setattr(
        ert_server_controller, "_ERT_SERVER_EXECUTABLE_FILE", server_script
    )
    nonexistent = str(tmp_path / "does_not_exist")
    proc = _DummyService(storage_path=nonexistent)
    proc.start()
    try:
        with pytest.raises(RuntimeError, match="No storage exists at"):
            proc.fetch_connection_info()
    finally:
        proc.shutdown()


def test_cleanup_service_files(tmpdir):
    with tmpdir.as_cwd():
        storage_service_name = "storage"
        storage_service_file = Path(f"{storage_service_name}_server.json")
        storage_service_file.write_text("storage_service info", encoding="utf-8")
        assert storage_service_file.exists()
        SERVICE_CONF_PATHS.add(tmpdir / storage_service_file)

        with pytest.raises(OSError, match="Signal 99 received"):
            cleanup_service_files(signum=99, frame=None)

        assert not storage_service_file.exists()
