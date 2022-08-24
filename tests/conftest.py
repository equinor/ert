import os
import resource
import shutil
from unittest.mock import MagicMock

import pkg_resources
import pytest
from hypothesis import HealthCheck, settings

from ert._c_wrappers.enkf import ResConfig
from ert.shared.services import Storage

from .utils import SOURCE_DIR

# CI runners produce unreliable test timings
# so too_slow healthcheck and deadline has to
# be supressed to avoid flaky behavior
settings.register_profile(
    "ci", max_examples=10, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)


@pytest.fixture(scope="session")
def source_root():
    return SOURCE_DIR


@pytest.fixture(scope="class")
def class_source_root(request):
    request.cls.SOURCE_ROOT = SOURCE_DIR
    request.cls.TESTDATA_ROOT = SOURCE_DIR / "test-data"
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


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        skip_quick = pytest.mark.skip(
            reason="skipping quick performance tests on --runslow"
        )
        for item in items:
            if "quick_only" in item.keywords:
                item.add_marker(skip_quick)
            if "ert3" in str(item.fspath):
                item.add_marker(pytest.mark.ert3)
            if item.get_closest_marker("requires_eclipse") and not config.getoption(
                "--eclipse_simulator"
            ):
                item.add_marker(pytest.mark.skip("Requires eclipse"))

    else:
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
            if "ert3" in str(item.fspath):
                item.add_marker(pytest.mark.ert3)
            if item.get_closest_marker("requires_eclipse") and not config.getoption(
                "--eclipse-simulator"
            ):
                item.add_marker(pytest.mark.skip("Requires eclipse"))
