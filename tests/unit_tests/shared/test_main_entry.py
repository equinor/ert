import logging
import sys
from unittest.mock import MagicMock

import pytest

import ert.__main__ as main


def test_main_logging(monkeypatch, caplog):
    parser_mock = MagicMock()
    parser_mock.func.side_effect = ValueError("This is a test")
    monkeypatch.setattr(logging.config, "dictConfig", MagicMock())
    monkeypatch.setattr(main, "ert_parser", MagicMock(return_value=parser_mock))
    monkeypatch.setattr(main, "start_ert_server", MagicMock())
    monkeypatch.setattr(main, "ErtPluginContext", MagicMock())
    monkeypatch.setattr(sys, "argv", ["ert", "test_run", "config.ert"])
    with pytest.raises(
        SystemExit, match='ERT crashed unexpectedly with "This is a test"'
    ):
        main.main()
    assert 'ERT crashed unexpectedly with "This is a test"' in caplog.text
    assert "Traceback" in caplog.text


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


def test_api_database_default(monkeypatch):
    monkeypatch.setattr(logging.config, "dictConfig", MagicMock())

    monkeypatch.setattr(main, "start_ert_server", MagicMock())
    monkeypatch.setattr(main, "ErtPluginContext", MagicMock())
    mocked_start_server = MagicMock()
    monkeypatch.setattr(
        "ert.services.storage_service.BaseService.start_server",
        mocked_start_server,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["ert", "api"],
    )

    main.main()
    # We expect default value from Storage class, validate that no explicit
    # value is given for database_url
    mocked_start_server.assert_called_once_with(ert_config=None, verbose=True)


def test_api_database_url_forwarded(monkeypatch):
    monkeypatch.setattr(logging.config, "dictConfig", MagicMock())

    monkeypatch.setattr(main, "start_ert_server", MagicMock())
    monkeypatch.setattr(main, "ErtPluginContext", MagicMock())
    mocked_start_server = MagicMock()
    monkeypatch.setattr(
        "ert.services.storage_service.BaseService.start_server",
        mocked_start_server,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["ert", "api", "--database-url", "TEST_DATABASE_URL"],
    )

    main.main()
    mocked_start_server.assert_called_once_with(
        ert_config=None, database_url="TEST_DATABASE_URL", verbose=True
    )
