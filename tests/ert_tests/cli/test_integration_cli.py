import os
import shutil
import threading
import time
from argparse import ArgumentParser
from unittest.mock import MagicMock, Mock, call, patch

import pytest

import ert_shared
from ert_shared.cli import ENSEMBLE_SMOOTHER_MODE, TEST_RUN_MODE
from ert_shared.cli.main import run_cli, ErtCliError
from ert_shared.feature_toggling import FeatureToggling
from ert_shared.main import ert_parser


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
        parsed = ert_parser(
            parser,
            [
                ENSEMBLE_SMOOTHER_MODE,
                "--current-case",
                "test_case",
                "--target-case",
                "test_case",
                "poly_example/poly.ert",
                "--port-range",
                "1024-65535",
            ],
        )

        with pytest.raises(ErtCliError, match="They were both: test_case"):
            run_cli(parsed)


def test_runpath_file(tmpdir, source_root):
    shutil.copytree(
        os.path.join(source_root, "test-data", "local", "poly_example"),
        os.path.join(str(tmpdir), "poly_example"),
    )

    with tmpdir.as_cwd():
        with open("poly_example/poly.ert", "a") as fh:
            config_lines = [
                "LOAD_WORKFLOW_JOB ASSERT_RUNPATH_FILE\n"
                "LOAD_WORKFLOW TEST_RUNPATH_FILE\n",
                "HOOK_WORKFLOW TEST_RUNPATH_FILE PRE_SIMULATION\n",
            ]

            fh.writelines(config_lines)

        parser = ArgumentParser(prog="test_main")
        parsed = ert_parser(
            parser,
            [
                ENSEMBLE_SMOOTHER_MODE,
                "--target-case",
                "poly_runpath_file",
                "--realizations",
                "1,2,4,8,16,32,64",
                "poly_example/poly.ert",
                "--port-range",
                "1024-65535",
            ],
        )

        run_cli(parsed)

        assert os.path.isfile("RUNPATH_WORKFLOW_0.OK")
        assert os.path.isfile("RUNPATH_WORKFLOW_1.OK")


def test_ensemble_evaluator(tmpdir, source_root):
    shutil.copytree(
        os.path.join(source_root, "test-data", "local", "poly_example"),
        os.path.join(str(tmpdir), "poly_example"),
    )

    with tmpdir.as_cwd():
        parser = ArgumentParser(prog="test_main")
        parsed = ert_parser(
            parser,
            [
                ENSEMBLE_SMOOTHER_MODE,
                "--target-case",
                "poly_runpath_file",
                "--realizations",
                "1,2,4,8,16,32,64",
                "--disable-ensemble-evaluator",
                "poly_example/poly.ert",
                "--port-range",
                "1024-65535",
            ],
        )
        FeatureToggling.update_from_args(parsed)

        assert FeatureToggling.is_enabled("ensemble-evaluator") is False

        run_cli(parsed)
        FeatureToggling.reset()


def test_ensemble_evaluator_disable_monitoring(tmpdir, source_root):
    shutil.copytree(
        os.path.join(source_root, "test-data", "local", "poly_example"),
        os.path.join(str(tmpdir), "poly_example"),
    )

    with tmpdir.as_cwd():
        parser = ArgumentParser(prog="test_main")
        parsed = ert_parser(
            parser,
            [
                ENSEMBLE_SMOOTHER_MODE,
                "--disable-ensemble-evaluator",
                "--disable-monitoring",
                "--target-case",
                "poly_runpath_file",
                "--realizations",
                "1,2,4,8,16,32,64",
                "poly_example/poly.ert",
                "--port-range",
                "1024-65535",
            ],
        )
        FeatureToggling.update_from_args(parsed)

        assert FeatureToggling.is_enabled("ensemble-evaluator") is False

        run_cli(parsed)
        FeatureToggling.reset()


def test_cli_test_run(tmpdir, source_root, mock_cli_run):
    shutil.copytree(
        os.path.join(source_root, "test-data", "local", "poly_example"),
        os.path.join(str(tmpdir), "poly_example"),
    )

    with tmpdir.as_cwd():
        parser = ArgumentParser(prog="test_main")
        parsed = ert_parser(
            parser,
            [
                TEST_RUN_MODE,
                "poly_example/poly.ert",
                "--port-range",
                "1024-65535",
            ],
        )
        run_cli(parsed)

    monitor_mock, thread_join_mock, thread_start_mock = mock_cli_run
    monitor_mock.assert_called_once()
    thread_join_mock.assert_called_once()
    thread_start_mock.assert_has_calls([[call(), call()]])


def test_cli_test_connection_error(tmpdir, source_root, capsys):
    shutil.copytree(
        os.path.join(source_root, "test-data", "local", "poly_example"),
        os.path.join(str(tmpdir), "poly_example"),
    )

    def _simulate_connection_refused():
        time.sleep(1)  # there's always some waiting for a connection
        raise ConnectionRefusedError("Connection error")

    mock_monitor = MagicMock()
    mock_monitor.__enter__.side_effect = _simulate_connection_refused

    with patch(
        "ert_shared.status.tracker.evaluator.create_ee_monitor",
        return_value=mock_monitor,
    ), patch(
        "ert_shared.ensemble_evaluator.evaluator.ee_monitor.create",
        return_value=mock_monitor,
    ):
        with tmpdir.as_cwd():
            parser = ArgumentParser(prog="test_main")
            parsed = ert_parser(
                parser,
                [
                    TEST_RUN_MODE,
                    "poly_example/poly.ert",
                    "--port-range",
                    "1024-65535",
                ],
            )
            with pytest.raises(ErtCliError):
                run_cli(parsed)

                capture = capsys.readouterr()
                assert "Connection error" in capture.err
