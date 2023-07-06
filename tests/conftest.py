import fileinput
import os
import resource
import shutil
from argparse import ArgumentParser
from unittest.mock import MagicMock

import pkg_resources
import pytest
from hypothesis import HealthCheck, settings

from ert.__main__ import ert_parser
from ert._c_wrappers.enkf import EnKFMain, ErtConfig
from ert.cli import ENSEMBLE_EXPERIMENT_MODE
from ert.cli.main import run_cli
from ert.services import StorageService
from ert.shared.feature_toggling import FeatureToggling
from ert.storage import open_storage

from .utils import SOURCE_DIR

# Timeout settings are unreliable both on CI and
# when running pytest with xdist so we disable it
settings.register_profile(
    "no_timeouts",
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("no_timeouts")


@pytest.fixture(scope="session", name="source_root")
def fixture_source_root():
    return SOURCE_DIR


@pytest.fixture(scope="class")
def class_source_root(request, source_root):
    request.cls.SOURCE_ROOT = source_root
    request.cls.TESTDATA_ROOT = source_root / "test-data"
    request.cls.SHARE_ROOT = pkg_resources.resource_filename("ert.shared", "share")
    yield


@pytest.fixture(autouse=True)
def env_save():
    exceptions = [
        "PYTEST_CURRENT_TEST",
        "KMP_DUPLICATE_LIB_OK",
        "KMP_INIT_AT_FORK",
        "QT_API",
    ]
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


@pytest.fixture(name="setup_case")
def fixture_setup_case(tmp_path, source_root, monkeypatch):
    def copy_case(path, config_file):
        shutil.copytree(os.path.join(source_root, "test-data", path), "test_data")
        monkeypatch.chdir(tmp_path / "test_data")
        return ErtConfig.from_file(config_file)

    monkeypatch.chdir(tmp_path)
    yield copy_case


@pytest.fixture()
def poly_case(setup_case):
    return EnKFMain(setup_case("poly_example", "poly.ert"))


@pytest.fixture()
def snake_oil_case_storage(copy_snake_oil_case_storage, tmp_path, source_root):
    return EnKFMain(ErtConfig.from_file("snake_oil.ert"))


@pytest.fixture()
def snake_oil_case(setup_case):
    return EnKFMain(setup_case("snake_oil", "snake_oil.ert"))


@pytest.fixture()
def minimum_case(use_tmpdir):
    with open("minimum_config", "w", encoding="utf-8") as fout:
        fout.write(
            "NUM_REALIZATIONS 10\nQUEUE_OPTION LOCAL MAX_RUNNING 50\nMAX_RUNTIME 42"
        )
    return EnKFMain(ErtConfig.from_file("minimum_config"))


@pytest.fixture(name="copy_case")
def fixture_copy_case(tmp_path, source_root, monkeypatch):
    def _copy_case(path):
        shutil.copytree(os.path.join(source_root, "test-data", path), "test_data")
        monkeypatch.chdir(tmp_path / "test_data")

    monkeypatch.chdir(tmp_path)
    yield _copy_case


@pytest.fixture()
def copy_poly_case(copy_case):
    copy_case("poly_example")


@pytest.fixture()
def copy_snake_oil_surface(copy_case):
    copy_case("snake_oil_field")


@pytest.fixture()
def copy_snake_oil_case(copy_case):
    copy_case("snake_oil")


@pytest.fixture(
    name="copy_snake_oil_case_storage",
    params=[
        pytest.param(0, marks=pytest.mark.xdist_group(name="snake_oil_case_storage"))
    ],
)
def fixture_copy_snake_oil_case_storage(_shared_snake_oil_case, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    shutil.copytree(_shared_snake_oil_case, "test_data")
    monkeypatch.chdir("test_data")


@pytest.fixture()
def copy_minimum_case(copy_case):
    copy_case("simple_config")


@pytest.fixture()
def use_tmpdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


@pytest.fixture()
def mock_start_server(monkeypatch):
    start_server = MagicMock()
    monkeypatch.setattr(StorageService, "start_server", start_server)
    yield start_server


@pytest.fixture()
def mock_connect(monkeypatch):
    connect = MagicMock()
    monkeypatch.setattr(StorageService, "connect", connect)
    yield connect


@pytest.fixture(scope="session", autouse=True)
def hide_window(request):
    if request.config.getoption("--show-gui"):
        yield
        return

    old_value = os.environ.get("QT_QPA_PLATFORM")
    os.environ["QT_QPA_PLATFORM"] = "minimal"
    yield
    if old_value is None:
        del os.environ["QT_QPA_PLATFORM"]
    else:
        os.environ["QT_QPA_PLATFORM"] = old_value


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
    parser.addoption("--show-gui", action="store_true", default=False)


def pytest_collection_modifyitems(config, items):
    for item in items:
        fixtures = getattr(item, "fixturenames", ())
        if "qtbot" in fixtures or "qapp" in fixtures or "qtmodeltester" in fixtures:
            item.add_marker("requires_window_manager")

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


def _run_snake_oil(source_root):
    shutil.copytree(os.path.join(source_root, "test-data", "snake_oil"), "test_data")
    os.chdir("test_data")
    with fileinput.input("snake_oil.ert", inplace=True) as fin:
        for line_nr, line in enumerate(fin):
            if line_nr == 1:
                print("QUEUE_OPTION LOCAL MAX_RUNNING 5", end="")
            if "NUM_REALIZATIONS 25" in line:
                print("NUM_REALIZATIONS 5", end="")
            else:
                print(line, end="")

        parser = ArgumentParser(prog="test_main")
        parsed = ert_parser(
            parser,
            [
                ENSEMBLE_EXPERIMENT_MODE,
                "--current-case",
                "default_0",
                "snake_oil.ert",
            ],
        )
        FeatureToggling.update_from_args(parsed)

        run_cli(parsed)
        FeatureToggling.reset()


@pytest.fixture
def _shared_snake_oil_case(request, monkeypatch, source_root):
    """This fixture will run the snake_oil case to populate storage,
    this is quite slow, but the results will be cached. If something comes
    out of sync, clear the cache and start again.
    """
    snake_path = request.config.cache.mkdir(
        "snake_oil_data" + os.environ.get("PYTEST_XDIST_WORKER", "")
    )
    monkeypatch.chdir(snake_path)
    if not os.listdir(snake_path):
        _run_snake_oil(source_root)
    else:
        monkeypatch.chdir("test_data")

    yield os.getcwd()


@pytest.fixture
def storage(tmp_path):
    with open_storage(tmp_path, mode="w") as storage:
        yield storage


@pytest.fixture
def new_ensemble(storage):
    experiment_id = storage.create_experiment()
    return storage.create_ensemble(
        experiment_id, name="new_ensemble", ensemble_size=100
    )


@pytest.fixture
def snake_oil_storage(snake_oil_case_storage):
    with open_storage(snake_oil_case_storage.ert_config.ens_path, mode="w") as storage:
        yield storage


@pytest.fixture
def snake_oil_default_storage(snake_oil_case_storage):
    with open_storage(snake_oil_case_storage.resConfig().ens_path) as storage:
        yield storage.get_ensemble_by_name("default_0")
