import logging
import os
from functools import partial
from unittest.mock import MagicMock, patch

import pytest

import everest
from ert.config import QueueSystem
from everest.bin.everest_script import everest_entry
from everest.bin.kill_script import kill_entry
from everest.bin.monitor_script import monitor_entry
from everest.config import EverestConfig, ServerConfig
from everest.detached import (
    ExperimentState,
    everserver_status,
    update_everserver_status,
)
from tests.everest.utils import capture_streams

CONFIG_FILE_MINIMAL = "config_minimal.yml"


def run_detached_monitor_mock(status=ExperimentState.completed, error=None, **kwargs):
    path = os.path.join(
        os.getcwd(), "everest_output/detached_node_output/.session/status"
    )
    update_everserver_status(path, status, message=error)


@patch("everest.bin.everest_script.run_detached_monitor")
@patch("everest.bin.everest_script.wait_for_server")
@patch("everest.bin.everest_script.start_server")
@patch(
    "everest.bin.everest_script.everserver_status",
    return_value={"status": ExperimentState.never_run, "message": None},
)
@patch("everest.bin.everest_script.start_experiment")
def test_everest_entry_debug(
    start_experiment_mock,
    everserver_status_mock,
    start_server_mock,
    wait_for_server_mock,
    start_monitor_mock,
    caplog,
    tmp_path,
    copy_math_func_test_data_to_tmp,
):
    """Test running everest with --debug"""

    # Need to deactivate the logging.config.dictConfig() statement in the entry
    # point for the caplog fixture to be able to catch logs:
    logger_conf = tmp_path / "dummy_logger.conf"
    logger_conf.write_text("", encoding="utf-8")
    with (
        patch("everest.bin.utils.LOGGING_CONFIG", str(logger_conf)),
        caplog.at_level(logging.DEBUG),
    ):
        everest_entry([CONFIG_FILE_MINIMAL, "--debug", "--skip"])
    logstream = "\n".join(caplog.messages)
    start_server_mock.assert_called_once()
    wait_for_server_mock.assert_called_once()
    start_monitor_mock.assert_called_once()
    everserver_status_mock.assert_called()
    start_experiment_mock.assert_called_once()

    # the config file itself is dumped at DEBUG level
    assert '"controls"' in logstream
    assert '"objective_functions"' in logstream
    assert '"name": "distance"' in logstream
    assert f'"config_path": "{os.getcwd()}/config_minimal.yml"' in logstream


@patch("everest.bin.everest_script.run_detached_monitor")
@patch("everest.bin.everest_script.wait_for_server")
@patch("everest.bin.everest_script.start_server")
@patch(
    "everest.bin.everest_script.everserver_status",
    return_value={"status": ExperimentState.never_run, "message": None},
)
@patch("everest.bin.everest_script.start_experiment")
def test_everest_entry(
    start_experiment_mock,
    everserver_status_mock,
    start_server_mock,
    wait_for_server_mock,
    start_monitor_mock,
    copy_math_func_test_data_to_tmp,
):
    """Test running everest in detached mode"""
    everest_entry([CONFIG_FILE_MINIMAL, "--skip"])
    start_server_mock.assert_called_once()
    wait_for_server_mock.assert_called_once()
    start_monitor_mock.assert_called_once()
    everserver_status_mock.assert_called()
    start_experiment_mock.assert_called_once()


@patch("everest.bin.everest_script.server_is_running", return_value=False)
@patch("everest.bin.everest_script.run_detached_monitor")
@patch("everest.bin.everest_script.wait_for_server")
@patch("everest.bin.everest_script.start_server")
@patch(
    "everest.bin.everest_script.everserver_status",
    return_value={"status": ExperimentState.completed, "message": None},
)
@patch("everest.bin.everest_script.start_experiment")
def test_everest_entry_detached_already_run(
    start_experiment_mock,
    everserver_status_mock,
    start_server_mock,
    wait_for_server_mock,
    start_monitor_mock,
    server_is_running_mock,
    copy_math_func_test_data_to_tmp,
):
    """Test everest detached, when an optimization has already run"""
    # optimization already run, notify the user
    with capture_streams() as (out, _):
        everest_entry([CONFIG_FILE_MINIMAL, "--skip-prompt"])
    assert "--new-run" in out.getvalue()
    start_server_mock.assert_not_called()
    start_monitor_mock.assert_not_called()
    wait_for_server_mock.assert_not_called()
    server_is_running_mock.assert_called_once()
    everserver_status_mock.assert_called()
    everserver_status_mock.reset_mock()

    # stopping the server has no effect
    kill_entry([CONFIG_FILE_MINIMAL])
    everserver_status_mock.assert_not_called()

    # forcefully re-run the case
    everest_entry([CONFIG_FILE_MINIMAL, "--new-run", "--skip-prompt"])
    start_server_mock.assert_called_once()
    start_monitor_mock.assert_called_once()
    everserver_status_mock.assert_called()


@patch("everest.bin.monitor_script.server_is_running", return_value=False)
@patch("everest.bin.monitor_script.run_detached_monitor")
@patch(
    "everest.bin.monitor_script.everserver_status",
    return_value={"status": ExperimentState.completed, "message": None},
)
def test_everest_entry_detached_already_run_monitor(
    everserver_status_mock,
    start_monitor_mock,
    server_is_running_mock,
    copy_math_func_test_data_to_tmp,
):
    """Test everest detached, when an optimization has already run"""
    # optimization already run, notify the user
    with capture_streams() as (out, _):
        monitor_entry([CONFIG_FILE_MINIMAL])
    assert "--new-run" in out.getvalue()
    start_monitor_mock.assert_not_called()
    server_is_running_mock.assert_called_once()
    everserver_status_mock.assert_called()
    everserver_status_mock.reset_mock()


@patch("everest.bin.everest_script.server_is_running", return_value=True)
@patch("everest.bin.kill_script.server_is_running", return_value=True)
@patch("everest.bin.everest_script.run_detached_monitor")
@patch("everest.bin.everest_script.wait_for_server")
@patch("everest.bin.everest_script.start_server")
@patch("everest.bin.kill_script.stop_server", return_value=True)
@patch("everest.bin.kill_script.wait_for_server_to_stop")
@patch(
    "everest.bin.everest_script.everserver_status",
    return_value={"status": ExperimentState.completed, "message": None},
)
def test_everest_entry_detached_running(
    everserver_status_mock,
    wait_for_server_to_stop_mock,
    stop_server_mock,
    start_server_mock,
    wait_for_server_mock,
    start_monitor_mock,
    server_is_running_mock_kill_script,
    server_is_running_mock_everest_script,
    copy_math_func_test_data_to_tmp,
):
    """Test everest detached, optimization is running"""
    # can't start a new run if one is already running
    with capture_streams() as (out, _):
        everest_entry([CONFIG_FILE_MINIMAL, "--new-run", "--skip-prompt"])
    assert "everest kill" in out.getvalue()
    assert "everest monitor" in out.getvalue()
    start_server_mock.assert_not_called()
    start_monitor_mock.assert_not_called()
    wait_for_server_mock.assert_not_called()
    server_is_running_mock_everest_script.assert_called_once()
    server_is_running_mock_everest_script.reset_mock()
    everserver_status_mock.assert_called_once()
    everserver_status_mock.reset_mock()

    # stop the server
    kill_entry([CONFIG_FILE_MINIMAL])
    stop_server_mock.assert_called_once()
    wait_for_server_to_stop_mock.assert_called_once()
    wait_for_server_mock.assert_not_called()
    server_is_running_mock_kill_script.assert_called_once()
    server_is_running_mock_kill_script.reset_mock()
    everserver_status_mock.assert_not_called()

    # if already running, nothing happens
    assert "everest kill" in out.getvalue()
    assert "everest monitor" in out.getvalue()
    everest_entry([CONFIG_FILE_MINIMAL, "--skip-prompt"])
    start_server_mock.assert_not_called()
    server_is_running_mock_everest_script.assert_called_once()
    everserver_status_mock.assert_called()


@patch("everest.bin.monitor_script.server_is_running", return_value=True)
@patch("everest.bin.monitor_script.run_detached_monitor")
@patch(
    "everest.bin.monitor_script.everserver_status",
    return_value={"status": ExperimentState.completed, "message": None},
)
def test_everest_entry_detached_running_monitor(
    everserver_status_mock,
    start_monitor_mock,
    server_is_running_mock,
    copy_math_func_test_data_to_tmp,
):
    """Test everest detached, optimization is running, monitoring"""
    # Attach to a running optimization.
    with capture_streams():
        monitor_entry([CONFIG_FILE_MINIMAL])
    start_monitor_mock.assert_called_once()
    server_is_running_mock.assert_called_once()
    everserver_status_mock.assert_called()


@patch("everest.bin.monitor_script.server_is_running", return_value=False)
@patch("everest.bin.monitor_script.run_detached_monitor")
@patch(
    "everest.bin.monitor_script.everserver_status",
    return_value={"status": ExperimentState.completed, "message": None},
)
def test_everest_entry_monitor_no_run(
    everserver_status_mock,
    start_monitor_mock,
    server_is_running_mock,
    copy_math_func_test_data_to_tmp,
):
    """Test everest detached, optimization is running, monitoring"""
    # Attach to a running optimization.
    with capture_streams() as (out, _):
        monitor_entry([CONFIG_FILE_MINIMAL])
    assert "everest run" in out.getvalue()
    start_monitor_mock.assert_not_called()
    server_is_running_mock.assert_called_once()
    everserver_status_mock.assert_called()


@pytest.fixture(autouse=True)
def mock_ssl(monkeypatch):
    monkeypatch.setattr(everest.detached.client, "ssl", MagicMock())


@patch(
    "everest.bin.everest_script.run_detached_monitor",
    side_effect=partial(
        run_detached_monitor_mock,
        status=ExperimentState.failed,
        error="Reality was ripped to shreds!",
    ),
)
@patch("everest.bin.everest_script.wait_for_server")
@patch("everest.bin.everest_script.start_server")
@patch("everest.bin.everest_script.start_experiment")
def test_exception_raised_when_server_run_fails(
    start_experiment_mock,
    start_server_mock,
    wait_for_server_mock,
    start_monitor_mock,
    copy_math_func_test_data_to_tmp,
):
    with pytest.raises(SystemExit, match="Reality was ripped to shreds!"):
        everest_entry([CONFIG_FILE_MINIMAL, "--skip-prompt"])


@patch("everest.bin.monitor_script.server_is_running", return_value=True)
@patch(
    "everest.bin.monitor_script.run_detached_monitor",
    side_effect=partial(
        run_detached_monitor_mock,
        status=ExperimentState.failed,
        error="Reality was ripped to shreds!",
    ),
)
def test_exception_raised_when_server_run_fails_monitor(
    start_monitor_mock, server_is_running_mock, copy_math_func_test_data_to_tmp
):
    with pytest.raises(SystemExit, match="Reality was ripped to shreds!"):
        monitor_entry([CONFIG_FILE_MINIMAL])


@patch(
    "everest.bin.everest_script.run_detached_monitor",
    side_effect=run_detached_monitor_mock,
)
@patch("everest.bin.everest_script.wait_for_server")
@patch("everest.bin.everest_script.start_server")
@patch("everest.bin.everest_script.start_experiment")
def test_complete_status_for_normal_run(
    start_experiment_mock,
    start_server_mock,
    wait_for_server_mock,
    start_monitor_mock,
    copy_math_func_test_data_to_tmp,
):
    everest_entry([CONFIG_FILE_MINIMAL, "--skip-prompt"])
    config = EverestConfig.load_file(CONFIG_FILE_MINIMAL)
    status_path = ServerConfig.get_everserver_status_path(config.output_dir)
    status = everserver_status(status_path)
    expected_status = ExperimentState.completed
    expected_error = None

    assert expected_status == status["status"]
    assert expected_error == status["message"]


@patch("everest.bin.monitor_script.server_is_running", return_value=True)
@patch(
    "everest.bin.monitor_script.run_detached_monitor",
    side_effect=run_detached_monitor_mock,
)
def test_complete_status_for_normal_run_monitor(
    start_monitor_mock, server_is_running_mock, copy_math_func_test_data_to_tmp
):
    monitor_entry([CONFIG_FILE_MINIMAL])
    config = EverestConfig.load_file(CONFIG_FILE_MINIMAL)
    status_path = ServerConfig.get_everserver_status_path(config.output_dir)
    status = everserver_status(status_path)
    expected_status = ExperimentState.completed
    expected_error = None

    assert expected_status == status["status"]
    assert expected_error == status["message"]


class ServerStatus:
    pass


@pytest.mark.parametrize(
    "server_queue_system, simulator_queue_system",
    [
        (QueueSystem.LSF, QueueSystem.LSF),
        (QueueSystem.SLURM, QueueSystem.LSF),
        (QueueSystem.LOCAL, QueueSystem.LSF),
        (QueueSystem.LSF, QueueSystem.SLURM),
        (QueueSystem.SLURM, QueueSystem.SLURM),
        (QueueSystem.LOCAL, QueueSystem.SLURM),
        (QueueSystem.LOCAL, QueueSystem.LOCAL),
    ],
)
def test_that_run_everest_prints_where_it_runs(
    server_queue_system, simulator_queue_system, capsys, change_to_tmpdir
):
    EverestConfig.with_defaults(
        simulator={"queue_system": {"name": simulator_queue_system}},
        server={"queue_system": {"name": server_queue_system}},
    ).dump("config.yml")

    with (
        patch(
            "everest.bin.everest_script.EverestStorage.check_for_deprecated_seba_storage"
        ),
        patch(
            "everest.bin.everest_script.ServerConfig.get_everserver_status_path",
            return_value="mock_status_path",
        ),
        patch(
            "everest.bin.everest_script.everserver_status",
            return_value={"status": ExperimentState.never_run, "message": None},
        ),
        patch("everest.bin.everest_script.server_is_running", return_value=False),
        patch("everest.bin.everest_script.start_server"),
        patch("everest.bin.everest_script.wait_for_server"),
        patch("everest.bin.everest_script.start_experiment"),
    ):
        everest_entry(["config.yml", "--skip-prompt"])

        captured = capsys.readouterr().out

        server_loc_str = (
            "this machine"
            if server_queue_system == QueueSystem.LOCAL
            else "the " + server_queue_system + " queue."
        )
        expected_server_str = (
            f"* The optimization will be run by an experiment "
            f"server on {server_loc_str}"
        )

        expected_simulator_str = (
            "The experiment server will submit the ERT forward model to run on "
        ) + (
            "this machine"
            if simulator_queue_system == QueueSystem.LOCAL
            else f"the {simulator_queue_system} queue."
        )

        assert "=======You are now running everest=======" in captured
        assert "* Monitoring from this machine:" in captured
        assert expected_server_str in captured
        assert expected_simulator_str in captured
