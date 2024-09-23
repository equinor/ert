import fileinput
import logging
import os
import resource
import shutil
import sys
from argparse import ArgumentParser
from pathlib import Path
from unittest.mock import MagicMock

from qtpy.QtWidgets import QApplication

from _ert.threading import set_signal_handler

if sys.version_info >= (3, 9):
    from importlib.resources import files
else:
    from importlib_resources import files

import pytest
from hypothesis import HealthCheck, settings
from hypothesis import strategies as st
from qtpy.QtCore import QDir

import _ert.forward_model_runner.cli
from ert.__main__ import ert_parser
from ert.cli.main import run_cli
from ert.config import ErtConfig
from ert.ensemble_evaluator.config import EvaluatorServerConfig
from ert.mode_definitions import ENSEMBLE_EXPERIMENT_MODE, ES_MDA_MODE
from ert.services import StorageService
from ert.storage import open_storage

from .utils import SOURCE_DIR

st.register_type_strategy(Path, st.builds(Path, st.text().map(lambda x: "/tmp/" + x)))


@pytest.fixture(autouse=True)
def no_jobs_file_retry(monkeypatch):
    monkeypatch.setattr(_ert.forward_model_runner.cli, "JOBS_JSON_RETRY_TIME", 0)


@pytest.fixture(autouse=True)
def log_check():
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)
    yield
    logger_after = logging.getLogger()
    level_after = logger_after.getEffectiveLevel()
    assert (
        level_after == logging.WARNING
    ), f"Detected differences in log environment: Changed to {level_after}"


@pytest.fixture(scope="session", autouse=True)
def _reraise_thread_exceptions_on_main_thread():
    """Allow `_ert.threading.ErtThread` to re-raise exceptions on main thread"""
    set_signal_handler()


@pytest.fixture
def _qt_add_search_paths(qapp):
    "Ensure that icons and such are found by the tests"
    QDir.addSearchPath("img", str(files("ert.gui").joinpath("resources/gui/img")))


# Timeout settings are unreliable both on CI and
# when running pytest with xdist so we disable it
settings.register_profile(
    "no_timeouts",
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
    print_blob=True,
)
settings.load_profile("no_timeouts")


@pytest.fixture()
def set_site_config(monkeypatch, tmp_path):
    test_site_config = tmp_path / "test_site_config.ert"
    test_site_config.write_text("JOB_SCRIPT job_dispatch.py\nQUEUE_SYSTEM LOCAL\n")
    monkeypatch.setenv("ERT_SITE_CONFIG", str(test_site_config))


@pytest.fixture(scope="session", name="source_root")
def fixture_source_root():
    return SOURCE_DIR


@pytest.fixture(scope="class")
def class_source_root(request, source_root):
    request.cls.SOURCE_ROOT = source_root
    request.cls.TESTDATA_ROOT = source_root / "test-data" / "ert"
    yield


@pytest.fixture(autouse=True)
def env_save():
    exceptions = [
        "PYTEST_CURRENT_TEST",
        "KMP_DUPLICATE_LIB_OK",
        "KMP_INIT_AT_FORK",
        "QT_API",
        "COV_CORE_CONTEXT",
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
def fixture_setup_case(tmp_path_factory, source_root, monkeypatch):
    def copy_case(path, config_file):
        tmp_path = tmp_path_factory.mktemp(path.replace("/", "-"))
        shutil.copytree(
            os.path.join(source_root, "test-data/ert", path), tmp_path / "test_data"
        )
        monkeypatch.chdir(tmp_path / "test_data")
        return ErtConfig.from_file(config_file)

    yield copy_case


@pytest.fixture()
def poly_case(setup_case):
    return setup_case("poly_example", "poly.ert")


@pytest.fixture()
def snake_oil_case_storage(copy_snake_oil_case_storage):
    return ErtConfig.from_file("snake_oil.ert")


@pytest.fixture()
def heat_equation_storage(copy_heat_equation_storage):
    return ErtConfig.from_file("config.ert")


@pytest.fixture()
def snake_oil_case(setup_case):
    return setup_case("snake_oil", "snake_oil.ert")


@pytest.fixture()
def minimum_case(use_tmpdir):
    with open("minimum_config", "w", encoding="utf-8") as fout:
        fout.write(
            "NUM_REALIZATIONS 10\nQUEUE_OPTION LOCAL MAX_RUNNING 50\nMAX_RUNTIME 42"
        )
    return ErtConfig.from_file("minimum_config")


@pytest.fixture(name="copy_case")
def fixture_copy_case(tmp_path_factory, source_root, monkeypatch):
    def _copy_case(path):
        tmp_path = tmp_path_factory.mktemp(path.replace("/", "-"))
        shutil.copytree(
            os.path.join(source_root, "test-data/ert", path),
            tmp_path / "test_data",
            ignore=shutil.ignore_patterns("storage"),
        )
        monkeypatch.chdir(tmp_path / "test_data")

    yield _copy_case


@pytest.fixture()
def copy_poly_case(copy_case):
    copy_case("poly_example")


@pytest.fixture()
def copy_snake_oil_field(copy_case):
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


@pytest.fixture
def copy_heat_equation_storage(_shared_heat_equation, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    shutil.copytree(_shared_heat_equation, "heat_equation")
    monkeypatch.chdir("heat_equation")


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
    if sys.platform == "darwin":
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
    else:
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
    parser.addoption(
        "--openpbs",
        action="store_true",
        default=False,
        help="Run OpenPBS tests against the real cluster",
    )
    parser.addoption(
        "--lsf",
        action="store_true",
        default=False,
        help="Run LSF tests against the real cluster.",
    )
    parser.addoption(
        "--slurm",
        action="store_true",
        default=False,
        help="Run Slurm tests against a real cluster.",
    )
    parser.addoption("--show-gui", action="store_true", default=False)


@pytest.fixture
def _qt_excepthook(monkeypatch):
    """Hook into Python's unhandled exception handler and quit Qt if it's
    running. This will prevent a stall in the event that a Python exception
    occurs inside a Qt slot.

    """
    next_excepthook = sys.excepthook

    def excepthook(cls, exc, tb):
        if app := QApplication.instance():
            app.quit()
        next_excepthook(cls, exc, tb)

    monkeypatch.setattr(sys, "excepthook", excepthook)


def pytest_collection_modifyitems(config, items):
    for item in items:
        fixtures = getattr(item, "fixturenames", ())
        if "qtbot" in fixtures or "qapp" in fixtures or "qtmodeltester" in fixtures:
            item.add_marker("requires_window_manager")

    # Override Python's excepthook on all "requires_window_manager" tests
    for item in items:
        if item.get_closest_marker("requires_window_manager"):
            item.fixturenames.append("_qt_excepthook")
            item.fixturenames.append("_qt_add_search_paths")

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
    shutil.copytree(
        os.path.join(source_root, "test-data/ert", "snake_oil"), "test_data"
    )
    os.chdir("test_data")
    with fileinput.input("snake_oil.ert", inplace=True) as fin:
        for line in fin:
            if "NUM_REALIZATIONS 25" in line:
                print("NUM_REALIZATIONS 5", end="")
            else:
                print(line, end="")

    parser = ArgumentParser(prog="test_main")
    parsed = ert_parser(
        parser,
        [
            ENSEMBLE_EXPERIMENT_MODE,
            "--disable-monitor",
            "--current-ensemble",
            "default_0",
            "snake_oil.ert",
        ],
    )

    run_cli(parsed)


def _run_heat_equation(source_root):
    shutil.copytree(
        os.path.join(source_root, "test-data", "ert", "heat_equation"), "test_data"
    )
    os.chdir("test_data")

    parser = ArgumentParser(prog="test_main")
    parsed = ert_parser(
        parser,
        [
            ES_MDA_MODE,
            "--disable-monitor",
            "config.ert",
        ],
    )

    run_cli(parsed)


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
    if not os.path.exists(snake_path / "test_data"):
        _run_snake_oil(source_root)
    else:
        monkeypatch.chdir("test_data")

    yield os.getcwd()


@pytest.fixture
def _shared_heat_equation(request, monkeypatch, source_root):
    """This fixture will run the heat_equation case to populate storage,
    this is quite slow, but the results will be cached. If something comes
    out of sync, clear the cache and start again.
    """
    snake_path = request.config.cache.mkdir(
        "heat_equation_data" + os.environ.get("PYTEST_XDIST_WORKER", "")
    )
    monkeypatch.chdir(snake_path)
    if not os.listdir(snake_path):
        _run_heat_equation(source_root)
    else:
        monkeypatch.chdir("test_data")

    yield os.getcwd()


@pytest.fixture
def storage(tmp_path):
    with open_storage(tmp_path / "storage", mode="w") as storage:
        yield storage


@pytest.fixture
def new_ensemble(storage):
    experiment_id = storage.create_experiment()
    return storage.create_ensemble(
        experiment_id, name="new_ensemble", ensemble_size=100
    )


@pytest.fixture
def snake_oil_storage(snake_oil_case_storage):
    with open_storage(snake_oil_case_storage.ens_path, mode="w") as storage:
        yield storage


@pytest.fixture
def snake_oil_default_storage(snake_oil_case_storage):
    with open_storage(snake_oil_case_storage.ens_path) as storage:
        experiment = storage.get_experiment_by_name("ensemble-experiment")
        yield experiment.get_ensemble_by_name("default_0")


@pytest.fixture(scope="session")
def block_storage_path(source_root):
    path = source_root / "test-data/ert/block_storage/snake_oil"
    if not path.is_dir():
        pytest.skip(
            "'test-data/ert/block_storage' has not been checked out.\n"
            "Make sure you have git-lfs installed and run: "
            "git submodule update --init --recursive"
        )
    return path.parent


@pytest.fixture(autouse=True)
def no_cert_in_test(monkeypatch):
    # Do not generate certificates during test, parts of it can be time
    # consuming (e.g. 30 seconds)
    # Specifically generating the RSA key <_openssl.RSA_generate_key_ex>
    class MockESConfig(EvaluatorServerConfig):
        def __init__(self, *args, **kwargs):
            if "use_token" not in kwargs:
                kwargs["use_token"] = False
            if "generate_cert" not in kwargs:
                kwargs["generate_cert"] = False
            super().__init__(*args, **kwargs)

    monkeypatch.setattr("ert.cli.main.EvaluatorServerConfig", MockESConfig)
