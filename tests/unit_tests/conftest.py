# This conftest still exists so that tests files can import ert_utils
import logging
import os
import sys

import pytest

from ert.ensemble_evaluator.config import EvaluatorServerConfig


@pytest.fixture(autouse=True)
def log_check():
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)
    yield
    logger_after = logging.getLogger()
    level_after = logger_after.getEffectiveLevel()
    assert (
        logging.WARNING == level_after
    ), f"Detected differences in log environment: Changed to {level_after}"


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


@pytest.fixture(scope="session", autouse=True)
def ensure_bin_in_path():
    """
    Running pytest directly without enabling a virtualenv is perfectly valid.
    However, our tests assume that `job_dispatch.py` is in PATH which it may not be.
    This fixture prepends the path to the current Python for tests to pass when not
    in a virtualenv.
    """
    path = os.environ["PATH"]
    exec_path = os.path.dirname(sys.executable)
    os.environ["PATH"] = exec_path + ":" + path
