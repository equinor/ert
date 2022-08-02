# This conftest still exists so that tests files can import ert_utils
import logging
import pytest


@pytest.fixture(autouse=False)
def log_check():
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)
    yield
    logger_after = logging.getLogger()
    level_after = logger_after.getEffectiveLevel()
    assert (
        logging.WARNING == level_after
    ), f"Detected differences in log environment: Changed to {level_after}"
