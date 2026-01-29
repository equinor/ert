import importlib
import os
import signal
import sys
import threading
import time
from pathlib import Path
from textwrap import dedent

import pytest

from ert.services import ert_server
from ert.services.ert_server import (
    _ERT_SERVER_CONNECTION_INFO_FILE,
    SERVICE_CONF_PATHS,
    ErtServer,
    ServerBootFail,
    cleanup_service_files,
)


def local_exec_args(script_args: str | list[str]) -> list[str]:
    """
    Convenience function that returns the exec_args for executing a Python
    script in the directory of old '_base_service.py'.

    This is done instead of using 'python -m [module path]' due to the '-m' flag
    adding the user's current working directory to sys.path. Executing a Python
    script by itself will add the directory of the script rather than the
    current working directory, thus we avoid accidentally importing user's
    directories that just happen to have the same names as the ones we use.
    """
    if isinstance(script_args, str):
        script = script_args
        rest: list[str] = []
    else:
        script = script_args[0]
        rest = script_args[1:]
    script = f"_{script}_main.py"

    services_spec = importlib.util.find_spec("ert.services")
    if services_spec is None or services_spec.origin is None:
        raise RuntimeError("Cannot find module ert.services")

    # PS: origin points to the __init__.py file
    services_folder = Path(services_spec.origin).parent

    return [sys.executable, str(services_folder / script), *rest]


class _DummyService(ErtServer):
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
    monkeypatch.setattr(ert_server, "_ERT_SERVER_EXECUTABLE_FILE", server_script)
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
time.sleep(30)
sys.exit(2)
"""
)
def test_not_respond(server):
    server._timeout = 1
    with pytest.raises(TimeoutError):
        server.fetch_connection_info()
    assert server.shutdown() == -signal.SIGTERM


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
    _BaseService is responsible for deleting the JSON file after the
    subprocess is finished running.
    """
    server.fetch_connection_info()  # wait for it to start
    time.sleep(2)  # ensure subprocess is done before calling shutdown()

    assert not os.path.exists(_ERT_SERVER_CONNECTION_INFO_FILE)


@pytest.mark.script(
    """\
os.write(fd, b'{"authtoken": "test123", "urls": ["url"]}')
os.close(fd)
time.sleep(10) # ensure "server" doesn't exit before test
"""
)
def test_singleton_start(monkeypatch, server_script, tmp_path):
    monkeypatch.setattr(ert_server, "_ERT_SERVER_EXECUTABLE_FILE", server_script)
    with _DummyService.start_server(".", timeout=10) as service:
        assert service.wait_until_ready()
        assert (tmp_path / _ERT_SERVER_CONNECTION_INFO_FILE).exists()

    assert not (tmp_path / _ERT_SERVER_CONNECTION_INFO_FILE).exists()


@pytest.mark.slow
@pytest.mark.script(
    """\
time.sleep(1)
os.write(fd, b'{"authtoken": "test123", "urls": ["url"]}')
os.close(fd)
"""
)
def test_singleton_connect(monkeypatch, tmp_path, server_script):
    monkeypatch.setattr(ert_server, "_ERT_SERVER_EXECUTABLE_FILE", server_script)
    with _DummyService().start_server(".", timeout=10) as server:
        client = _DummyService.connect(project=tmp_path, timeout=30)
        assert server is client


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
                self.client = _DummyService.connect(project=tmp_path, timeout=30)
            except Exception as ex:
                self.exception = ex
            ready_event.set()

    client_thread = ClientThread()
    client_thread.start()

    start_event.wait()  # Client thread has started
    monkeypatch.setattr(ert_server, "_ERT_SERVER_EXECUTABLE_FILE", server_script)
    with _DummyService.start_server(".") as server:
        ready_event.wait()  # Client thread has connected to server
        assert not getattr(client_thread, "exception", None), (
            f"Exception from connect: {client_thread.exception}"
        )
        client = client_thread.client
        assert client is not server
        assert client.fetch_connection_info() == server.fetch_connection_info()

    assert not (tmp_path / _ERT_SERVER_CONNECTION_INFO_FILE).exists()


@pytest.mark.parametrize(
    ("script", "should_exist"), [("storage", True), ("foobar", False)]
)
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
        storage_service_file = Path(f"{storage_service_name}_server.json")
        storage_service_file.write_text("storage_service info", encoding="utf-8")
        assert storage_service_file.exists()
        SERVICE_CONF_PATHS.add(tmpdir / storage_service_file)

        webviz_service_name = "webviz-ert"
        webviz_service_file = Path(f"{webviz_service_name}_server.json")
        webviz_service_file.write_text("webviz-ert info", encoding="utf-8")
        assert webviz_service_file.exists()
        SERVICE_CONF_PATHS.add(tmpdir / webviz_service_file)

        with pytest.raises(OSError, match="Signal 99 received"):
            cleanup_service_files(signum=99, frame=None)

        assert not storage_service_file.exists()
        assert not webviz_service_file.exists()
