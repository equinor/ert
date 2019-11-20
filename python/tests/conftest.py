import pytest
import unittest
import resource
import functools
import os

def source_root():
    path_list = os.path.dirname(os.path.abspath(__file__)).split("/")
    while len(path_list) > 0:
        git_path = os.path.join(os.sep, "/".join(path_list), ".git")
        if os.path.isdir(git_path):
            return os.path.join(os.sep, *path_list)
        path_list.pop()
    raise RuntimeError('Cannot find the source folder')


def has_equinor_test_data():
    return os.path.isdir(os.path.join(source_root(), "test-data", "Equinor"))


def pytest_runtest_setup(item):
    if item.get_closest_marker("equinor_test") and not has_equinor_test_data():
        pytest.skip("Test requires Equinor data")


@pytest.fixture(autouse=True)
def env_save():
    environment_pre = [(key, val) for key, val in os.environ.items() if key != "PYTEST_CURRENT_TEST"]
    yield
    environment_post = [(key, val) for key, val in os.environ.items() if key != "PYTEST_CURRENT_TEST"]
    if set(environment_pre) != set(environment_post):
        raise EnvironmentError("Your environment has changed after that test, please reset")
