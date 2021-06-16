import os
from pathlib import Path

import pytest


def source_dir():
    src = Path("@CMAKE_CURRENT_SOURCE_DIR@/../..")
    if src.is_dir():
        return src.relative_to(Path.cwd())

    # If the file was not correctly configured by cmake, look for the source
    # folder, assuming the build folder is inside the source folder.
    test_path = Path(__file__)
    while test_path != Path("/"):
        if (test_path / ".git").is_dir():
            return test_path
        test_path = test_path.parent
    raise RuntimeError("Cannot find the source folder")


def has_equinor_test_data():
    return os.path.isdir(os.path.join(source_dir(), "test-data", "Equinor"))


def pytest_runtest_setup(item):
    if item.get_closest_marker("equinor_test") and not has_equinor_test_data():
        pytest.skip("Test requires Equinor data")


def pytest_configure(config):
    config.addinivalue_line("markers", "equinor_test")


@pytest.fixture(autouse=True)
def env_save():
    environment_pre = [
        (key, val) for key, val in os.environ.items() if key != "PYTEST_CURRENT_TEST"
    ]
    yield
    environment_post = [
        (key, val) for key, val in os.environ.items() if key != "PYTEST_CURRENT_TEST"
    ]
    if set(environment_pre) != set(environment_post):
        raise EnvironmentError(
            "Your environment has changed after that test, please reset"
        )


@pytest.fixture
def pathlib_source_root():
    return source_dir()


@pytest.fixture(scope="class")
def class_pathlib_source_root(request):
    SOURCE_ROOT = source_dir()

    request.cls.SOURCE_ROOT = SOURCE_ROOT
    request.cls.TESTDATA_ROOT = SOURCE_ROOT / "test-data"
    request.cls.SHARE_ROOT = SOURCE_ROOT / "share"
    request.cls.EQUINOR_DATA = (request.cls.TESTDATA_ROOT / "Equinor").is_symlink()
    yield
