from tests import libres_source_root
import pytest
import unittest
import resource
import functools
import os


@pytest.fixture
def source_root():
    return libres_source_root()


def has_equinor_test_data():
    return os.path.isdir(os.path.join(libres_source_root(), "test-data", "Equinor"))


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
