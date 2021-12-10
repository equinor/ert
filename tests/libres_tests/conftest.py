import os
from pathlib import Path

import pytest


@pytest.fixture()
def setup_tmpdir(tmpdir):
    with tmpdir.as_cwd():
        yield


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
