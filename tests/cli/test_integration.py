import sys
import pytest
import os
import shutil
import threading
import ert_shared
from ert_shared.main import ert_parser
from argparse import ArgumentParser
from ert_shared.cli.main import run_cli
from ert_shared.cli import ENSEMBLE_SMOOTHER_MODE, TEST_RUN_MODE

if sys.version_info >= (3, 3):
    from unittest.mock import Mock
else:
    from mock import Mock


@pytest.fixture()
def mock_cli_run(monkeypatch):
    mocked_monitor = Mock()
    mocked_thread_start = Mock()
    mocked_thread_join = Mock()
    monkeypatch.setattr(threading.Thread, "start", mocked_thread_start)
    monkeypatch.setattr(threading.Thread, "join", mocked_thread_join)
    monkeypatch.setattr(ert_shared.cli.monitor.Monitor, "monitor", mocked_monitor)
    yield mocked_monitor, mocked_thread_join, mocked_thread_start


def test_target_case_equal_current_case(tmpdir, source_root):
    shutil.copytree(
        os.path.join(source_root, "test-data", "local", "poly_example"),
        os.path.join(str(tmpdir), "poly_example"),
    )
    with tmpdir.as_cwd():
        parser = ArgumentParser(prog="test_main")
        parsed = ert_parser(parser, [ENSEMBLE_SMOOTHER_MODE, "--current-case", 'test_case',
                                    "--target-case", 'test_case',
                                    'poly_example/poly.ert'])

        with pytest.raises(SystemExit):
            run_cli(parsed)


def test_cli_test_run(tmpdir, source_root, mock_cli_run):
    shutil.copytree(
        os.path.join(source_root, "test-data", "local", "poly_example"),
        os.path.join(str(tmpdir), "poly_example"),
    )

    with tmpdir.as_cwd():
        parser = ArgumentParser(prog="test_main")
        parsed = ert_parser(parser, [TEST_RUN_MODE, "poly_example/poly.ert"])
        run_cli(parsed)

    monitor_mock, thread_join_mock, thread_start_mock = mock_cli_run
    monitor_mock.assert_called_once()
    thread_join_mock.assert_called_once()
    thread_start_mock.assert_called_once()
