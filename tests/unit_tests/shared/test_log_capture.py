import logging

import pytest

from ert.run_models.base_run_model import captured_logs

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "log_level, expect_propagation",
    [
        (logger.debug, False),
        (logger.info, False),
        (logger.warning, False),
        (logger.error, True),
        (logger.critical, True),
    ],
)
def test_default_log_capture(log_level, expect_propagation, caplog):
    with caplog.at_level(logging.INFO):
        with captured_logs() as logs:
            log_level("This is not actually an error")
        if expect_propagation:
            assert "This is not actually an error" in logs.messages
        else:
            assert "This is not actually an error" not in logs.messages


@pytest.mark.parametrize(
    "log_level",
    [
        logging.DEBUG,
        logging.INFO,
        logging.INFO,
        logging.ERROR,
        logging.CRITICAL,
    ],
)
@pytest.mark.parametrize(
    "log_func, corresponding_level",
    [
        (logger.debug, logging.DEBUG),
        (logger.info, logging.INFO),
        (logger.warning, logging.WARNING),
        (logger.error, logging.ERROR),
        (logger.critical, logging.CRITICAL),
    ],
)
def test_custom_log_capture(log_level, log_func, corresponding_level, caplog):
    with caplog.at_level(logging.DEBUG):
        with captured_logs(level=log_level) as logs:
            log_func("This is a message")
        if corresponding_level >= logs.level:
            assert "This is a message" in logs.messages
        else:
            assert "This is a message" not in logs.messages
