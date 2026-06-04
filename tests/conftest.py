import os
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from ert.config import ErtConfig
from ert.plugins import ErtRuntimePlugins


def source_dir() -> Path:
    src = Path("@CMAKE_CURRENT_SOURCE_DIR@/../..")
    if src.is_dir():
        return src.relative_to(Path.cwd())

    # If the file was not correctly configured by cmake, look for the source
    # folder, assuming the build folder is inside the source folder.
    current_path = Path(__file__)
    while current_path != Path("/"):
        if (current_path / ".git").is_dir():
            return current_path
        # This is to find root dir for git worktrees
        if (current_path / ".git").is_file():
            with (current_path / ".git").open(encoding="utf-8") as f:
                for line in f:
                    if "gitdir:" in line:
                        return current_path

        current_path = current_path.parent
    raise RuntimeError("Cannot find the source folder")


SOURCE_DIR: Path = source_dir()


def pytest_addoption(parser):
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


def pytest_collection_modifyitems(config, items):
    for item in items:
        fixtures = getattr(item, "fixturenames", ())
        if "qtbot" in fixtures or "qapp" in fixtures or "qtmodeltester" in fixtures:
            item.add_marker("requires_window_manager")
        if "unused_tcp_port" in fixtures:
            item.add_marker(pytest.mark.flaky(rerun=3))
        if any(
            f in fixtures
            for f in [
                "tmpdir",
                "use_tmpdir",
                "tmp_path",
                "tmp_path_factory",
            ]
        ):
            item.add_marker("creates_tmpdir")

        if any(f in item.keywords for f in ["flaky", "skip_mac_ci"]):
            item.add_marker("unreliable")

        if any(f in item.keywords for f in ["memory_test", "limit_memory"]):
            item.add_marker("high_utilization")

        # Override Python's excepthook on all "requires_window_manager" tests
        if item.get_closest_marker("requires_window_manager"):
            item.fixturenames.append("_qt_excepthook")
            item.fixturenames.append("_qt_add_search_paths")
        if item.get_closest_marker("requires_eclipse") and not config.getoption(
            "--eclipse-simulator"
        ):
            item.add_marker(pytest.mark.skip("Requires eclipse"))

        if "snapshot" in getattr(item, "fixturenames", ()):
            item.add_marker(pytest.mark.snapshot_test)

    if os.environ.get("ERT_TESTS_RUN_ON_MAC_CI"):
        for item in items:
            if "skip_mac_ci" in item.keywords:
                item.add_marker(pytest.mark.skip(reason="Skipped on mac ci"))


@pytest.fixture(autouse=True)
def print_system_load_on_test_failure(request):
    yield  # Run the actual test

    rep = getattr(request.node, "_rep_call", None)
    if rep and rep.failed:
        load1, load5, load15 = os.getloadavg()
        print(
            "System load after test failure (1/5/15min): "
            f"{load1:.2f}, {load5:.2f}, {load15:.2f}, cpu_count={os.cpu_count()}"
        )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Make the result of an individual test available in an item"""
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"_rep_{rep.when}", rep)


@pytest.fixture
def change_to_tmpdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


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
    set_pre_post = set(environment_pre).difference(set(environment_post))
    set_post_pre = set(environment_post).difference(set(environment_pre))
    assert len(set_xor) == 0, (
        f"Detected differences in environment: {set_xor}. "
        f"Pre-post: {set_pre_post}, post-pre: {set_post_pre}"
    )


@pytest.fixture
def use_site_configurations_with_no_site_logging():
    with patch("ert.__main__.setup_site_logging", lambda *args, **kwargs: None):
        yield


@pytest.fixture
def use_site_configurations_with_no_queue_options():
    def ErtRuntimePluginsWithNoQueueOptions(**kwargs):
        return ErtRuntimePlugins(**(kwargs | {"queue_options": None}))

    with patch(
        "ert.plugins.plugin_manager.ErtRuntimePlugins",
        ErtRuntimePluginsWithNoQueueOptions,
    ):
        yield


@pytest.fixture(scope="session", name="source_root")
def fixture_source_root():
    return SOURCE_DIR


@pytest.fixture(name="setup_case")
def fixture_setup_case(tmp_path_factory, source_root, monkeypatch):
    def copy_case(path, config_file):
        tmp_path = tmp_path_factory.mktemp(path.replace("/", "-"))
        shutil.copytree(
            Path(source_root) / "test-data" / "ert" / path,
            tmp_path / "test_data",
        )
        monkeypatch.chdir(tmp_path / "test_data")
        return ErtConfig.from_file(config_file)

    return copy_case


@pytest.fixture
def poly_case(setup_case):
    return setup_case("poly_example", "poly.ert")
