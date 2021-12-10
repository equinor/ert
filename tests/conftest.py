import os

import pkg_resources
import pytest
from hypothesis import HealthCheck, settings

from utils import SOURCE_DIR

# CI runners produce unreliable test timings
# so too_slow healthcheck and deadline has to
# be supressed to avoid flaky behavior
settings.register_profile(
    "ci", max_examples=10, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)


@pytest.fixture(scope="session")
def source_root():
    return SOURCE_DIR


@pytest.fixture(scope="class")
def class_source_root(request):
    request.cls.SOURCE_ROOT = SOURCE_DIR
    request.cls.TESTDATA_ROOT = SOURCE_DIR / "test-data"
    request.cls.SHARE_ROOT = pkg_resources.resource_filename("ert_shared", "share")
    request.cls.EQUINOR_DATA = (request.cls.TESTDATA_ROOT / "Equinor").is_symlink()
    yield


@pytest.fixture(autouse=True)
def env_save():
    exceptions = ["PYTEST_CURRENT_TEST", "KMP_DUPLICATE_LIB_OK", "KMP_INIT_AT_FORK"]
    environment_pre = [
        (key, val) for key, val in os.environ.items() if key not in exceptions
    ]
    yield
    environment_post = [
        (key, val) for key, val in os.environ.items() if key not in exceptions
    ]
    set_xor = set(environment_pre).symmetric_difference(set(environment_post))
    assert len(set_xor) == 0, f"Detected differences in environment: {set_xor}"
