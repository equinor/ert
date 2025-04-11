import logging
import sys
from unittest.mock import MagicMock

import filelock
import pytest

import ert
import ert.__main__ as main


@pytest.mark.usefixtures("use_tmpdir")
def test_main_logging(monkeypatch, caplog):
    parser_mock = MagicMock()
    parser_mock.func.side_effect = ValueError("This is a test")
    monkeypatch.setattr(logging.config, "dictConfig", MagicMock())
    monkeypatch.setattr(main, "ert_parser", MagicMock(return_value=parser_mock))
    monkeypatch.setattr(main, "ErtPluginContext", MagicMock())
    monkeypatch.setattr(sys, "argv", ["ert", "test_run", "config.ert"])
    with pytest.raises(
        SystemExit, match='ERT crashed unexpectedly with "This is a test"'
    ):
        main.main()
    assert 'ERT crashed unexpectedly with "This is a test"' in caplog.text
    assert "Traceback" in caplog.text


@pytest.mark.usefixtures("use_tmpdir")
def test_main_logging_argparse(monkeypatch, caplog):
    monkeypatch.setattr(logging.config, "dictConfig", MagicMock())
    monkeypatch.setattr(main, "valid_file", MagicMock(return_value=True))
    monkeypatch.setattr(main, "run_cli", MagicMock())
    monkeypatch.setattr(main, "ErtPluginContext", MagicMock())
    monkeypatch.setattr(sys, "argv", ["ert", "test_run", "config.ert"])
    with caplog.at_level(logging.INFO):
        main.main()
    assert "mode='test_run'" in caplog.text


@pytest.mark.usefixtures("copy_poly_case")
def test_storage_exception_is_not_unexpected_error(monkeypatch, caplog):
    file_lock_mock = MagicMock()
    caplog.set_level(logging.ERROR)

    def mock_acquire(*args, **kwargs):
        raise filelock.Timeout

    file_lock_mock.acquire = mock_acquire
    monkeypatch.setattr(ert.storage.local_storage, "FileLock", file_lock_mock)

    monkeypatch.setattr(sys, "argv", ["ert", "test_run", "poly.ert"])
    with pytest.raises(SystemExit) as exc_info:
        main.main()
    assert "ERT crashed unexpectedly" not in str(exc_info.value)
    assert "Failed to open storage" in str(exc_info.value)
