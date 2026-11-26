import logging
import sys
from argparse import Namespace
from pathlib import Path
from typing import override
from unittest import mock
from unittest.mock import MagicMock

import filelock
import pytest

import ert.__main__ as main
from ert import LoginProvider
from ert.cli.main import ErtCliError
from ert.plugins import ErtRuntimePlugins


@pytest.mark.usefixtures("use_tmpdir")
def test_that_ert_login_without_login_provider_exits_with_message(monkeypatch):
    monkeypatch.setattr(logging.config, "dictConfig", MagicMock())
    monkeypatch.setattr(main, "get_site_plugins", ErtRuntimePlugins)
    monkeypatch.setattr(sys, "argv", ["ert", "login"])
    with pytest.raises(SystemExit, match="No login provider is installed"):
        main.main()


def test_that_ert_login_calls_login_on_installed_login_provider():
    login_calls = []

    class MyLoginProvider(LoginProvider):
        @override
        def login(self) -> None:
            login_calls.append(True)

    main.run_login(Namespace(), ErtRuntimePlugins(login_provider=MyLoginProvider()))
    assert login_calls == [True]


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (RuntimeError("boom"), "Login with FailingLoginProvider failed: boom"),
        (KeyboardInterrupt(), "Login cancelled"),
    ],
)
def test_that_ert_login_turns_login_provider_exceptions_into_cli_errors(error, message):
    class FailingLoginProvider(LoginProvider):
        @override
        def login(self) -> None:
            raise error

    with pytest.raises(ErtCliError, match=message):
        main.run_login(
            Namespace(), ErtRuntimePlugins(login_provider=FailingLoginProvider())
        )


@pytest.mark.usefixtures("use_tmpdir")
def test_main_logging(
    monkeypatch, caplog, use_site_configurations_with_no_site_logging
):
    parser_mock = MagicMock()
    parser_mock.func.side_effect = ValueError("This is a test")
    monkeypatch.setattr(logging.config, "dictConfig", MagicMock())
    monkeypatch.setattr(main, "ert_parser", MagicMock(return_value=parser_mock))
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
    monkeypatch.setattr(sys, "argv", ["ert", "test_run", "config.ert"])
    with caplog.at_level(logging.INFO):
        main.main()
    assert "mode='test_run'" in caplog.text


@pytest.mark.usefixtures("copy_poly_case")
def test_storage_exception_is_not_unexpected_error(caplog):
    file_lock_mock = MagicMock()
    caplog.set_level(logging.ERROR)

    def mock_acquire(*args, **kwargs):
        raise filelock.Timeout("storage.lock")

    file_lock_mock.acquire = mock_acquire

    with (
        mock.patch("ert.storage.local_storage.FileLock", return_value=file_lock_mock),
        mock.patch.object(sys, "argv", ["ert", "test_run", "poly.ert"]),
        pytest.raises(SystemExit) as exc_info,
    ):
        main.main()
    assert "ERT crashed unexpectedly" not in str(exc_info.value)
    assert "Failed to open storage" in str(exc_info.value)
    assert "Not able to acquire lock" in str(exc_info.value)


def test_non_writable_log_directory_exits_with_message(monkeypatch, use_tmpdir):
    logs_dir = "logs_dir_without_write_access"
    Path(logs_dir).mkdir()
    Path(logs_dir).chmod(0o444)  # Read only access mode

    expected_exit_messages = [
        "Could not configure log handler for files.",
        "Check if you have write-access to the logs-directory",
        logs_dir,
    ]

    class ErtparserMock(MagicMock):
        logdir = logs_dir

    monkeypatch.setattr(main, "ert_parser", MagicMock(return_value=ErtparserMock()))

    with pytest.raises(SystemExit) as exc_info:
        main.main()
    assert all(
        expected_exit_message in str(exc_info)
        for expected_exit_message in expected_exit_messages
    )
