import asyncio
import contextlib
import os
import resource
import shutil
import threading
from functools import partial
from pathlib import Path
from unittest.mock import MagicMock

import pkg_resources
import pytest
import websockets
from hypothesis import HealthCheck, settings

from ert._c_wrappers.enkf import EnKFMain, ResConfig
from ert.shared.ensemble_evaluator.client import Client
from ert.shared.services import Storage

# CI runners produce unreliable test timings
# so too_slow healthcheck and deadline has to
# be supressed to avoid flaky behavior
settings.register_profile(
    "ci", max_examples=10, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)


def source_dir():
    src = Path("@CMAKE_CURRENT_SOURCE_DIR@/../..")
    if src.is_dir():
        return src.relative_to(Path.cwd())

    # If the file was not correctly configured by cmake, look for the source
    # folder, assuming the build folder is inside the source folder.
    current_path = Path(__file__)
    while current_path != Path("/"):
        if (current_path / ".git").is_dir():
            return current_path
        current_path = current_path.parent
    raise RuntimeError("Cannot find the source folder")


@pytest.fixture(scope="session")
def source_root():
    return source_dir()


@pytest.fixture(scope="class")
def class_source_root(request, source_root):
    request.cls.SOURCE_ROOT = source_root
    request.cls.TESTDATA_ROOT = source_root / "test-data"
    request.cls.SHARE_ROOT = pkg_resources.resource_filename("ert.shared", "share")
    request.cls.EQUINOR_DATA = (request.cls.TESTDATA_ROOT / "Equinor").is_symlink()
    yield


@pytest.fixture(autouse=True)
def env_save():
    exceptions = ["PYTEST_CURRENT_TEST", "KMP_DUPLICATE_LIB_OK", "KMP_INIT_AT_FORK"]
    environment_pre = [
        (key, val) for key, val in os.environ.items() if key not in exceptions
    ]
    yield
    environment_post = [
        (key, val) for key, val in os.environ.items() if key not in exceptions
    ]
    set_xor = set(environment_pre).symmetric_difference(set(environment_post))
    assert len(set_xor) == 0, f"Detected differences in environment: {set_xor}"


@pytest.fixture(scope="session", autouse=True)
def maximize_ulimits():
    """
    Bumps the soft-limit for max number of files up to its max-value
    since we know that the tests may open lots of files simultaneously.
    Resets to original when session ends.
    """
    limits = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (limits[1], limits[1]))
    yield
    resource.setrlimit(resource.RLIMIT_NOFILE, limits)


@pytest.fixture()
def setup_case(tmpdir, source_root):
    def copy_case(path, config_file):
        shutil.copytree(os.path.join(source_root, "test-data", path), "test_data")
        os.chdir("test_data")
        return ResConfig(config_file)

    with tmpdir.as_cwd():
        yield copy_case


@pytest.fixture()
def poly_example(setup_case):
    return EnKFMain(setup_case("local/poly_example", "poly.ert"))


@pytest.fixture()
def snake_oil_example(setup_case):
    return EnKFMain(setup_case("local/snake_oil", "snake_oil.ert"))


@pytest.fixture()
def minimum_example(setup_case):
    return EnKFMain(setup_case("local/simple_config", "minimum_config"))


@pytest.fixture()
def copy_case(tmpdir, source_root):
    def _copy_case(path):
        shutil.copytree(os.path.join(source_root, "test-data", path), "test_data")
        os.chdir("test_data")

    with tmpdir.as_cwd():
        yield _copy_case


@pytest.fixture()
def use_tmpdir(tmp_path):
    cwd = os.getcwd()
    os.chdir(tmp_path)
    yield
    os.chdir(cwd)


@pytest.fixture()
def mock_start_server(monkeypatch):
    connect_or_start_server = MagicMock()
    monkeypatch.setattr(Storage, "connect_or_start_server", connect_or_start_server)
    yield connect_or_start_server


def _mock_ws(host, port, messages, delay_startup=0):
    loop = asyncio.new_event_loop()
    done = loop.create_future()

    async def _handler(websocket, path):
        while True:
            msg = await websocket.recv()
            messages.append(msg)
            if msg == "stop":
                done.set_result(None)
                break

    async def _run_server():
        await asyncio.sleep(delay_startup)
        async with websockets.serve(_handler, host, port):
            await done

    loop.run_until_complete(_run_server())
    loop.close()


class MockWSMonitor:
    def __init__(self, unused_tcp_port) -> None:
        self.host = "localhost"
        self.url = f"ws://{self.host}:{unused_tcp_port}"
        self.messages = []
        self.mock_ws_thread = threading.Thread(
            target=partial(_mock_ws, messages=self.messages),
            args=(self.host, unused_tcp_port),
        )

    def start(self):
        self.mock_ws_thread.start()

    def join(self):
        with Client(self.url) as c:
            c.send("stop")
        self.mock_ws_thread.join()

    def join_and_get_messages(self):
        self.join()
        return self.messages


@pytest.fixture()
def mock_ws_monitor(unused_tcp_port):
    mock_ws_monitor = MockWSMonitor(unused_tcp_port=unused_tcp_port)
    mock_ws_monitor.start()
    yield mock_ws_monitor


@pytest.fixture
def mock_ws_thread():
    @contextlib.contextmanager
    def _mock_ws_thread(host, port, messages, delay_startup=0):
        mock_ws_thread = threading.Thread(
            target=partial(_mock_ws, messages=messages, delay_startup=delay_startup),
            args=(
                host,
                port,
            ),
        )
        mock_ws_thread.start()
        yield
        url = f"ws://{host}:{port}"
        with Client(url) as client:
            client.send("stop")
        mock_ws_thread.join()
        messages.pop()

    return _mock_ws_thread


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--eclipse-simulator",
        action="store_true",
        default=False,
        help="Defaults to not running tests that require eclipse.",
    )


@pytest.fixture
def setup_tmpdir(tmpdir):
    with tmpdir.as_cwd():
        yield


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        skip_quick = pytest.mark.skip(
            reason="skipping quick performance tests on --runslow"
        )
        for item in items:
            if "quick_only" in item.keywords:
                item.add_marker(skip_quick)
            if item.get_closest_marker("requires_eclipse") and not config.getoption(
                "--eclipse_simulator"
            ):
                item.add_marker(pytest.mark.skip("Requires eclipse"))

    else:
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
            if item.get_closest_marker("requires_eclipse") and not config.getoption(
                "--eclipse-simulator"
            ):
                item.add_marker(pytest.mark.skip("Requires eclipse"))
