import logging
import sys
from unittest.mock import MagicMock

import pytest

from ert_shared import main


def test_main_logging(monkeypatch, caplog):
    parser_mock = MagicMock()
    parser_mock.func.side_effect = ValueError
    monkeypatch.setattr(logging.config, "dictConfig", MagicMock())
    monkeypatch.setattr(main, "ert_parser", MagicMock(return_value=parser_mock))
    monkeypatch.setattr(main, "start_ert_server", MagicMock())
    monkeypatch.setattr(main, "ErtPluginContext", MagicMock())
    monkeypatch.setattr(sys, "argv", ["ert", "test_run", "config.ert"])
    with pytest.raises(SystemExit, match="ert crashed unexpectedly"):
        main.main()
    assert "ert crashed unexpectedly\nTraceback" in caplog.text


def test_main_logging_argparse(monkeypatch, caplog):
    monkeypatch.setattr(logging.config, "dictConfig", MagicMock())
    monkeypatch.setattr(main, "valid_file", MagicMock(return_value=True))
    monkeypatch.setattr(main, "run_cli", MagicMock())
    monkeypatch.setattr(main, "start_ert_server", MagicMock())
    monkeypatch.setattr(main, "ErtPluginContext", MagicMock())
    monkeypatch.setattr(sys, "argv", ["ert", "test_run", "config.ert"])
    with caplog.at_level(logging.INFO):
        main.main()
    assert "mode='test_run'" in caplog.text
