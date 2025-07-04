import os

import pytest


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
        if item.get_closest_marker("requires_eclipse") and not config.getoption(
            "--eclipse-simulator"
        ):
            item.add_marker(pytest.mark.skip("Requires eclipse"))

    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        skip_quick = pytest.mark.skip(
            reason="skipping quick performance tests on --runslow"
        )
        for item in items:
            if "quick_only" in item.keywords:
                item.add_marker(skip_quick)
    else:
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


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
    assert len(set_xor) == 0, f"Detected differences in environment: {set_xor}"
