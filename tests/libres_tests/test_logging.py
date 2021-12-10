import pytest
import logging

from res._lib import _test_logger


RECORDS = [
    ("res._test_logger", logging.DEBUG, "debug: foo"),
    ("res._test_logger", logging.INFO, "info: foo"),
    ("res._test_logger", logging.WARNING, "warning: foo"),
    ("res._test_logger", logging.ERROR, "error: foo"),
    ("res._test_logger", logging.CRITICAL, "critical: foo"),
]


@pytest.mark.parametrize(
    "level,records",
    [
        (logging.DEBUG, RECORDS),
        (logging.INFO, RECORDS[1:]),
        (logging.WARNING, RECORDS[2:]),
        (logging.ERROR, RECORDS[3:]),
        (logging.CRITICAL, RECORDS[4:]),
    ],
)
def test_logging_from_c(caplog, level, records):
    caplog.set_level(level)

    _test_logger("foo")
    assert caplog.record_tuples == records
