import logging
from unittest.mock import MagicMock
from ert.logging._log_util_abort import _log_util_abort


def test_log_util_abort(caplog, monkeypatch):
    shutdown_mock = MagicMock()
    monkeypatch.setattr(logging, "shutdown", shutdown_mock)
    with caplog.at_level(logging.ERROR):
        _log_util_abort("fname", 1, "some_func", "err_message", "my_backtrace")
    assert (
        "C trace:\nmy_backtrace \nwith message: err_message \nfrom file: "
        "fname in some_func at line: 1\n\nPython backtrace:"
    ) in caplog.text
    shutdown_mock.assert_called_once_with()  # must shutdown to propagate message
