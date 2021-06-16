import os

import pytest


def source_dir():
    src = "@CMAKE_CURRENT_SOURCE_DIR@/../.."
    if os.path.isdir(src):
        return os.path.realpath(src)

    # If the file was not correctly configured by cmake, look for the source
    # folder, assuming the build folder is inside the source folder.
    path_list = os.path.dirname(os.path.abspath(__file__)).split("/")
    while len(path_list) > 0:
        git_path = os.path.join(os.sep, "/".join(path_list), ".git")
        if os.path.isdir(git_path):
            return os.path.join(os.sep, *path_list)
        path_list.pop()
    raise RuntimeError("Cannot find the source folder")


@pytest.fixture
def source_root():
    return source_dir()


@pytest.fixture(scope="class")
def class_source_root(request):
    SOURCE_ROOT = source_dir()
    request.cls.TESTDATA_ROOT = os.path.join(SOURCE_ROOT, "test-data")
    request.cls.SHARE_ROOT = os.path.join(SOURCE_ROOT, "share")
    yield


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
