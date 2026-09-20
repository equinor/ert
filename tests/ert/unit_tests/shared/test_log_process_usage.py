import logging
import resource

from ert.__main__ import log_process_usage


def test_valid(caplog):
    caplog.set_level(logging.INFO)
    log_process_usage()

    assert len(caplog.records) == 1


def test_unsupported_platform(caplog, monkeypatch):
    """
    This test supposes that some exception occurs due to the current system not
    supporting the 'resource' package. The correct behaviour is to silently
    ignore errors because this is only meant for logging.
    """
    monkeypatch.delattr(resource, "getrusage")
    with caplog.at_level(logging.INFO):
        log_process_usage()

        # A warning has been logged but no exception has been raised
        assert len(caplog.records) == 1
        assert "Exception while trying to log" in caplog.records[0].message
