import os

import pytest

from utils import SOURCE_DIR


@pytest.fixture
def source_root():
    return SOURCE_DIR


@pytest.fixture(scope="class")
def class_source_root(request):
    request.cls.SOURCE_ROOT = SOURCE_DIR
    request.cls.TESTDATA_ROOT = SOURCE_DIR / "test-data"
    request.cls.SHARE_ROOT = SOURCE_DIR / "share"
    request.cls.EQUINOR_DATA = (request.cls.TESTDATA_ROOT / "Equinor").is_symlink()
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
        print(set(environment_pre) - set(environment_post))
        print(set(environment_post) - set(environment_pre))
        raise EnvironmentError(
            "Your environment has changed after that test, please reset"
        )
