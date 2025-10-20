import logging
import os
from functools import partial
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import everest
from ert.config import QueueSystem
from ert.run_models.everest_run_model import ExperimentStatus
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
@patch("everest.config.ServerConfig.get_server_context_from_conn_info")
@patch(
    "ert.services.StorageService.session",
    side_effect=[TimeoutError(), MagicMock()],
)
@patch(
    "everest.bin.everest_script.everserver_status",
    return_value={"status": ExperimentState.never_run, "message": None},
)
@patch("everest.bin.everest_script.start_experiment")
def test_everest_entry_debug(
    start_experiment_mock,
    everserver_status_mock,
    session_mock,
    get_server_context_from_conn_info_mock,
    start_server_mock,
    wait_for_server_mock,
    start_monitor_mock,
    caplog,
    change_to_tmpdir,
):
    """Test running everest with --debug"""

    Path("config.yml").touch()
    config = EverestConfig.with_defaults(config_path="./config.yml")
    config.dump("config.yml")

    # Need to deactivate the logging.config.dictConfig() statement in the entry
    # point for the caplog fixture to be able to catch logs:
    logger_conf = Path("dummy_logger.conf")
    logger_conf.write_text("", encoding="utf-8")
    with (
        patch("everest.bin.utils.LOGGING_CONFIG", str(logger_conf)),
        caplog.at_level(logging.DEBUG),
    ):
        everest_entry(["config.yml", "--debug", "--skip"])
    logstream = "\n".join(caplog.messages)
    start_server_mock.assert_called_once()
    wait_for_server_mock.assert_called_once()
    start_monitor_mock.assert_called_once()
    everserver_status_mock.assert_called()
    start_experiment_mock.assert_called_once()
    assert session_mock.call_count == 2
    assert get_server_context_from_conn_info_mock.call_count == 2

    # the config file itself is dumped at DEBUG level
    assert '"controls"' in logstream
    assert '"objective_functions"' in logstream
    assert '"name": "default"' in logstream
    assert f'"config_path": "{os.getcwd()}/config.yml"' in logstream


@patch("everest.bin.everest_script.run_detached_monitor")
@patch("everest.bin.everest_script.wait_for_server")
@patch("everest.bin.everest_script.start_server")
@patch("everest.config.ServerConfig.get_server_context_from_conn_info")
@patch(
    "ert.services.StorageService.session",
    side_effect=[TimeoutError(), MagicMock()],
)
@patch(
    "everest.bin.everest_script.everserver_status",
    return_value={"status": ExperimentState.never_run, "message": None},
)
@patch("everest.bin.everest_script.start_experiment")
def test_everest_entry(
    start_experiment_mock,
    everserver_status_mock,
    session_mock,
    get_server_context_from_conn_info_mock,
    start_server_mock,
    wait_for_server_mock,
    start_monitor_mock,
    change_to_tmpdir,
):
    """Test running everest in detached mode"""

    Path("config.yml").touch()
    config = EverestConfig.with_defaults(config_path="./config.yml")
    config.dump("config.yml")
    everest_entry(["config.yml", "--skip"])
    start_server_mock.assert_called_once()
    wait_for_server_mock.assert_called_once()
    start_monitor_mock.assert_called_once()
    everserver_status_mock.assert_called()
    start_experiment_mock.assert_called_once()
    assert session_mock.call_count == 2
    assert get_server_context_from_conn_info_mock.call_count == 2


@patch("everest.bin.everest_script.run_detached_monitor")
@patch("everest.bin.everest_script.wait_for_server")
@patch("everest.bin.everest_script.start_server")
@patch("everest.bin.everest_script.everserver_status")
@patch("everest.bin.everest_script.start_experiment")
@patch("everest.config.ServerConfig.get_server_context_from_conn_info")
@patch(
    "ert.services.StorageService.session",
    side_effect=[
        TimeoutError(),
        MagicMock(),
        TimeoutError(),
        TimeoutError(),
        MagicMock(),
    ],
)
def test_everest_entry_detached_already_run(
    session_mock,
    get_server_context_from_conn_info_mock,
    start_experiment_mock,
    everserver_status_mock,
    start_server_mock,
    wait_for_server_mock,
    start_monitor_mock,
    change_to_tmpdir,
):
    """Test everest detached, when an optimization has already run
    In this case we should just start a new run"""

    Path("config.yml").touch()
    config = EverestConfig.with_defaults(config_path="./config.yml")
    config.dump("config.yml")

    everserver_status_mock.return_value = {
        "status": ExperimentState.never_run,
        "message": None,
    }

    # start a new run
    everest_entry(["config.yml", "--skip-prompt"])
    start_server_mock.assert_called_once()
    start_monitor_mock.assert_called_once()
    everserver_status_mock.assert_called()
    start_experiment_mock.assert_called_once()
    assert session_mock.call_count == 2

    everserver_status_mock.reset_mock()
    start_server_mock.reset_mock()
    start_monitor_mock.reset_mock()
    start_experiment_mock.reset_mock()

    # stopping the server has no effect (not running)
    kill_entry(["config.yml"])
    everserver_status_mock.assert_not_called()
    assert session_mock.call_count == 3

    everserver_status_mock.return_value = {
        "status": ExperimentState.never_run,
        "message": None,
    }

    # run again, should start a new run like above
    everest_entry(["config.yml", "--skip-prompt"])
    start_server_mock.assert_called_once()
    start_monitor_mock.assert_called_once()
    everserver_status_mock.assert_called()
    start_experiment_mock.assert_called_once()
    assert session_mock.call_count == 5


@patch("everest.bin.monitor_script.run_detached_monitor")
@patch(
    "everest.bin.monitor_script.everserver_status",
    return_value={"status": ExperimentState.completed, "message": None},
)
@patch("ert.services.StorageService.session", side_effect=TimeoutError())
@patch("everest.config.ServerConfig.get_server_context_from_conn_info")
def test_everest_entry_detached_already_run_monitor(
    get_server_context_from_conn_info_mock,
    session_mock,
    everserver_status_mock,
    start_monitor_mock,
    change_to_tmpdir,
):
    """Test everest detached, when an optimization has already run"""

    Path("config.yml").touch()
    config = EverestConfig.with_defaults(config_path="./config.yml")
    config.dump("config.yml")

    # optimization already run, notify the user
    monitor_entry(["config.yml"])
    start_monitor_mock.assert_not_called()
    everserver_status_mock.assert_called()
    everserver_status_mock.reset_mock()
    get_server_context_from_conn_info_mock.assert_not_called()
    session_mock.assert_called_once()


@patch("ert.services.StorageService.session")
@patch("everest.config.ServerConfig.get_server_context_from_conn_info")
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
    get_server_context_from_conn_info_mock,
    session_mock,
    change_to_tmpdir,
):
    """Test everest detached, optimization is running"""

    Path("config.yml").touch()
    config = EverestConfig.with_defaults(config_path="./config.yml")
    config.dump("config.yml")

    # can't start a new run if one is already running
    with capture_streams() as (out, _):
        everest_entry(["config.yml", "--skip-prompt"])
    assert "everest kill" in out.getvalue()
    assert "everest monitor" in out.getvalue()
    start_server_mock.assert_not_called()
    start_monitor_mock.assert_not_called()
    wait_for_server_mock.assert_not_called()
    session_mock.assert_called_once()
    session_mock.reset_mock()
    get_server_context_from_conn_info_mock.assert_not_called()

    # stop the server
    kill_entry(["config.yml"])
    stop_server_mock.assert_called_once()
    wait_for_server_to_stop_mock.assert_called_once()
    session_mock.assert_called_once()
    session_mock.reset_mock()
    get_server_context_from_conn_info_mock.assert_called_once()
    wait_for_server_mock.assert_not_called()
    everserver_status_mock.assert_not_called()

    # if already running, nothing happens
    assert "everest kill" in out.getvalue()
    assert "everest monitor" in out.getvalue()
    everest_entry(["config.yml", "--skip-prompt"])
    session_mock.assert_called_once()
    start_server_mock.assert_not_called()


@patch("everest.bin.monitor_script.run_detached_monitor")
@patch(
    "everest.bin.monitor_script.everserver_status",
    return_value={"status": ExperimentState.completed, "message": None},
)
@patch("everest.config.ServerConfig.get_server_context_from_conn_info")
@patch("ert.services.StorageService.session")
def test_everest_entry_detached_running_monitor(
    session_mock,
    get_server_context_from_conn_info_mock,
    everserver_status_mock,
    start_monitor_mock,
    change_to_tmpdir,
):
    """Test everest detached, optimization is running, monitoring"""

    Path("config.yml").touch()
    config = EverestConfig.with_defaults(config_path="./config.yml")
    config.dump("config.yml")

    # Attach to a running optimization.
    with capture_streams():
        monitor_entry(["config.yml"])
    start_monitor_mock.assert_called_once()
    everserver_status_mock.assert_called()
    session_mock.assert_called_once()
    get_server_context_from_conn_info_mock.assert_called_once()


@patch("everest.bin.monitor_script.run_detached_monitor")
@patch(
    "everest.bin.monitor_script.get_experiment_status",
    return_value=ExperimentStatus(status=ExperimentState.completed),
)
@patch("everest.config.ServerConfig.get_server_context_from_conn_info")
@patch("ert.services.StorageService.session", side_effect=TimeoutError())
def test_everest_entry_monitor_already_run(
    session_mock,
    get_server_context_from_conn_info_mock,
    everserver_status_mock,
    start_monitor_mock,
    change_to_tmpdir,
):
    Path("config.yml").touch()
    config = EverestConfig.with_defaults(config_path="./config.yml")
    config.dump("config.yml")

    with capture_streams() as (out, _):
        monitor_entry(["config.yml"])
    assert "Optimization already completed." in out.getvalue()
    start_monitor_mock.assert_not_called()
    everserver_status_mock.assert_called()
    session_mock.assert_called_once()
    get_server_context_from_conn_info_mock.assert_not_called()


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
@patch("everest.config.ServerConfig.get_server_context_from_conn_info")
@patch(
    "ert.services.StorageService.session",
    side_effect=[TimeoutError(), MagicMock()],
)
def test_exception_raised_when_server_run_fails(
    session_mock,
    get_server_context_from_conn_info_mock,
    start_experiment_mock,
    start_server_mock,
    wait_for_server_mock,
    start_monitor_mock,
    change_to_tmpdir,
):
    Path("config.yml").touch()
    config = EverestConfig.with_defaults(config_path="./config.yml")
    config.dump("config.yml")

    with pytest.raises(SystemExit, match="Reality was ripped to shreds!"):
        everest_entry(["config.yml", "--skip-prompt"])


@patch(
    "everest.bin.monitor_script.run_detached_monitor",
    side_effect=partial(
        run_detached_monitor_mock,
        status=ExperimentState.failed,
        error="Reality was ripped to shreds!",
    ),
)
@patch("everest.config.ServerConfig.get_server_context_from_conn_info")
@patch("ert.services.StorageService.session")
def test_exception_raised_when_server_run_fails_monitor(
    session_mock,
    get_server_context_from_conn_info_mock,
    start_monitor_mock,
    change_to_tmpdir,
):
    Path("config.yml").touch()
    config = EverestConfig.with_defaults(config_path="./config.yml")
    config.dump("config.yml")

    with pytest.raises(SystemExit, match="Reality was ripped to shreds!"):
        monitor_entry(["config.yml"])


@patch(
    "everest.bin.everest_script.run_detached_monitor",
    side_effect=run_detached_monitor_mock,
)
@patch("everest.bin.everest_script.wait_for_server")
@patch("everest.bin.everest_script.start_server")
@patch("everest.bin.everest_script.start_experiment")
@patch(
    "ert.services.StorageService.session",
    side_effect=[TimeoutError(), MagicMock()],
)
@patch("everest.config.ServerConfig.get_server_context_from_conn_info")
def test_complete_status_for_normal_run(
    get_server_context_from_conn_info_mock,
    session_mock,
    start_experiment_mock,
    start_server_mock,
    wait_for_server_mock,
    start_monitor_mock,
    change_to_tmpdir,
):
    Path("config.yml").touch()
    config = EverestConfig.with_defaults(config_path="./config.yml")
    config.dump("config.yml")

    everest_entry(["config.yml", "--skip-prompt"])
    status_path = ServerConfig.get_everserver_status_path(config.output_dir)
    status = everserver_status(status_path)
    expected_status = ExperimentState.completed
    expected_error = None

    assert expected_status == status["status"]
    assert expected_error == status["message"]


@patch(
    "everest.bin.monitor_script.run_detached_monitor",
    side_effect=run_detached_monitor_mock,
)
@patch("ert.services.StorageService.session")
@patch("everest.config.ServerConfig.get_server_context_from_conn_info")
def test_complete_status_for_normal_run_monitor(
    get_server_context_from_conn_info_mock,
    session_mock,
    start_monitor_mock,
    change_to_tmpdir,
):
    Path("config.yml").touch()
    config = EverestConfig.with_defaults(config_path="./config.yml")
    config.dump("config.yml")
    monitor_entry(["config.yml"])
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
        patch(
            "ert.services.StorageService.session",
            side_effect=[TimeoutError(), MagicMock()],
        ),
        patch(
            "everest.config.ServerConfig.get_server_context_from_conn_info",
            return_value=("a", "b", ("c", "d")),
        ),
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
