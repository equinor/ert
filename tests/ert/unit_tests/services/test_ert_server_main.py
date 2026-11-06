import logging
import os
import sys
from unittest.mock import MagicMock

import pytest

from ert.services import _ert_server_main


def test_that_unexpected_server_crash_is_logged_with_traceback_and_reraised(
    monkeypatch, tmp_path, caplog
):
    monkeypatch.setattr(
        sys,
        "argv",
        ["ert_server", "--project", str(tmp_path), "--parent_pid", str(os.getppid())],
    )
    monkeypatch.setenv("ERT_STORAGE_TOKEN", "")
    monkeypatch.setattr(_ert_server_main.signal, "signal", MagicMock())
    monkeypatch.setattr(_ert_server_main.logging.config, "dictConfig", MagicMock())
    monkeypatch.setattr(_ert_server_main.uvicorn, "Config", MagicMock())
    monkeypatch.setattr(
        _ert_server_main, "_generate_certificate", lambda _: ("cert", "key", "pw")
    )
    monkeypatch.setattr(_ert_server_main, "setup_site_logging", MagicMock())
    monkeypatch.setattr(_ert_server_main, "terminate_on_parent_death", MagicMock())
    crash = RuntimeError("no free port")
    monkeypatch.setattr(_ert_server_main, "run_server", MagicMock(side_effect=crash))

    with (
        caplog.at_level(logging.ERROR, logger="ert.shared.storage.info"),
        pytest.raises(RuntimeError, match="no free port"),
    ):
        _ert_server_main.main()

    (record,) = caplog.records
    assert record.levelno == logging.ERROR
    assert record.message == "Ert server crashed unexpectedly"
    assert record.exc_info[1] is crash
