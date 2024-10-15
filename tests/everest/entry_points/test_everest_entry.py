import logging
import os
from functools import partial
from unittest.mock import PropertyMock, patch

import pytest

from ert.resources import all_shell_script_fm_steps
from everest.bin.everest_script import everest_entry
from everest.bin.kill_script import kill_entry
from everest.bin.monitor_script import monitor_entry
from everest.config import EverestConfig
from everest.detached import (
    SIM_PROGRESS_ENDPOINT,
    ServerStatus,
    everserver_status,
    update_everserver_status,
)
from everest.simulator import JOB_SUCCESS
from ieverest.bin.ieverest_script import ieverest_entry
from tests.everest.utils import capture_streams

CONFIG_FILE_MINIMAL = "config_minimal.yml"


def query_server_mock(cert, auth, endpoint):
    url = "localhost"
    sim_endpoint = "/".join([url, SIM_PROGRESS_ENDPOINT])

    def build_job(
        status=JOB_SUCCESS,
        start_time="begining",
        end_time="end",
        name="default_job",
        error=None,
    ):
        return {
            "status": status,
            "start_time": start_time,
            "end_time": end_time,
            "name": name,
            "error": error,
            "simulation": 0,
        }

    shell_cmd_jobs = [build_job(name=command) for command in all_shell_script_fm_steps]
    all_jobs = [
        *shell_cmd_jobs,
        build_job(name="make_pancakes"),
        build_job(name="make_scrambled_eggs"),
    ]
    if endpoint == sim_endpoint:
        return {
            "status": {
                "failed": 0,
                "running": 0,
                "complete": 1,
                "pending": 0,
                "waiting": 0,
            },
            "progress": [all_jobs],
            "batch_number": "0",
            "event": "end",
        }
    else:
        raise Exception("Stop! Hands in the air!")


def run_detached_monitor_mock(
    config, show_all_jobs=False, status=ServerStatus.completed, error=None
):
    update_everserver_status(config, status, message=error)


@patch("everest.bin.everest_script.run_detached_monitor")
@patch("everest.bin.everest_script.wait_for_server")
@patch("everest.bin.everest_script.start_server")
@patch(
    "everest.bin.everest_script.everserver_status",
    return_value={"status": ServerStatus.never_run, "message": None},
)
def test_everest_entry_debug(
    everserver_status_mock,
    start_server_mock,
    wait_for_server_mock,
    start_monitor_mock,
    caplog,
    copy_math_func_test_data_to_tmp,
):
    """Test running everest with --debug"""
    with caplog.at_level(logging.DEBUG):
        everest_entry([CONFIG_FILE_MINIMAL, "--debug"])
    logstream = "\n".join(caplog.messages)
    start_server_mock.assert_called_once()
    wait_for_server_mock.assert_called_once()
    start_monitor_mock.assert_called_once()
    everserver_status_mock.assert_called()

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
    return_value={"status": ServerStatus.never_run, "message": None},
)
def test_everest_entry(
    everserver_status_mock,
    start_server_mock,
    wait_for_server_mock,
    start_monitor_mock,
    copy_math_func_test_data_to_tmp,
):
    """Test running everest in detached mode"""
    everest_entry([CONFIG_FILE_MINIMAL])
    start_server_mock.assert_called_once()
    wait_for_server_mock.assert_called_once()
    start_monitor_mock.assert_called_once()
    everserver_status_mock.assert_called()


@patch("everest.bin.everest_script.server_is_running", return_value=False)
@patch("everest.bin.everest_script.run_detached_monitor")
@patch("everest.bin.everest_script.wait_for_server")
@patch("everest.bin.everest_script.start_server")
@patch(
    "everest.bin.everest_script.everserver_status",
    return_value={"status": ServerStatus.completed, "message": None},
)
def test_everest_entry_detached_already_run(
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
        everest_entry([CONFIG_FILE_MINIMAL])
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
    everest_entry([CONFIG_FILE_MINIMAL, "--new-run"])
    start_server_mock.assert_called_once()
    start_monitor_mock.assert_called_once()
    everserver_status_mock.assert_called()


@patch("everest.bin.monitor_script.server_is_running", return_value=False)
@patch("everest.bin.monitor_script.run_detached_monitor")
@patch(
    "everest.bin.monitor_script.everserver_status",
    return_value={"status": ServerStatus.completed, "message": None},
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
    return_value={"status": ServerStatus.completed, "message": None},
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
        everest_entry([CONFIG_FILE_MINIMAL, "--new-run"])
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
    everest_entry([CONFIG_FILE_MINIMAL])
    start_server_mock.assert_not_called()
    server_is_running_mock_everest_script.assert_called_once()
    everserver_status_mock.assert_called()


@patch("everest.bin.monitor_script.server_is_running", return_value=True)
@patch("everest.bin.monitor_script.run_detached_monitor")
@patch(
    "everest.bin.monitor_script.everserver_status",
    return_value={"status": ServerStatus.completed, "message": None},
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
    return_value={"status": ServerStatus.completed, "message": None},
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


@patch("everest.bin.everest_script.server_is_running", return_value=False)
@patch("everest.bin.everest_script.wait_for_server")
@patch("everest.bin.everest_script.start_server")
@patch("everest.detached._query_server", side_effect=query_server_mock)
@patch.object(
    EverestConfig,
    "server_context",
    new_callable=PropertyMock,
    return_value=("localhost", "", ""),
)
@patch("everest.detached.get_opt_status", return_value={})
@patch(
    "everest.bin.everest_script.everserver_status",
    return_value={"status": ServerStatus.never_run, "message": None},
)
def test_everest_entry_show_all_jobs(
    everserver_status_mock,
    get_opt_status_mock,
    get_server_context_mock,
    query_server_mock,
    start_server_mock,
    wait_for_server_mock,
    server_is_running_mock,
    copy_math_func_test_data_to_tmp,
):
    """Test running everest with --show-all-jobs"""

    # Test when --show-all-jobs flag is given shell command are in the list
    # of forward model jobs
    with capture_streams() as (out, _):
        everest_entry([CONFIG_FILE_MINIMAL, "--show-all-jobs"])
    for cmd in all_shell_script_fm_steps:
        assert cmd in out.getvalue()


@patch("everest.bin.everest_script.server_is_running", return_value=False)
@patch("everest.bin.everest_script.wait_for_server")
@patch("everest.bin.everest_script.start_server")
@patch("everest.detached._query_server", side_effect=query_server_mock)
@patch.object(
    EverestConfig,
    "server_context",
    new_callable=PropertyMock,
    return_value=("localhost", "", ""),
)
@patch("everest.detached.get_opt_status", return_value={})
@patch(
    "everest.bin.everest_script.everserver_status",
    return_value={"status": ServerStatus.never_run, "message": None},
)
def test_everest_entry_no_show_all_jobs(
    everserver_status_mock,
    get_opt_status_mock,
    get_server_context_mock,
    query_server_mock,
    start_server_mock,
    wait_for_server_mock,
    server_is_running_mock,
    copy_math_func_test_data_to_tmp,
):
    """Test running everest without --show-all-jobs"""

    # Test when --show-all-jobs flag is not given the shell command are not
    # in the list of forward model jobs
    with capture_streams() as (out, _):
        everest_entry([CONFIG_FILE_MINIMAL])
    for cmd in all_shell_script_fm_steps:
        assert cmd not in out.getvalue()

    # Check the other jobs are still there
    assert "make_pancakes" in out.getvalue()
    assert "make_scrambled_eggs" in out.getvalue()


@patch("everest.bin.monitor_script.server_is_running", return_value=True)
@patch("everest.detached._query_server", side_effect=query_server_mock)
@patch.object(
    EverestConfig,
    "server_context",
    new_callable=PropertyMock,
    return_value=("localhost", "", ""),
)
@patch("everest.detached.get_opt_status", return_value={})
@patch(
    "everest.bin.monitor_script.everserver_status",
    return_value={"status": ServerStatus.never_run, "message": None},
)
def test_monitor_entry_show_all_jobs(
    everserver_status_mock,
    get_opt_status_mock,
    get_server_context_mock,
    query_server_mock,
    server_is_running_mock,
    copy_math_func_test_data_to_tmp,
):
    """Test running everest with and without --show-all-jobs"""

    # Test when --show-all-jobs flag is given shell command are in the list
    # of forward model jobs

    with capture_streams() as (out, _):
        monitor_entry([CONFIG_FILE_MINIMAL, "--show-all-jobs"])
    for cmd in all_shell_script_fm_steps:
        assert cmd in out.getvalue()


@patch("everest.bin.monitor_script.server_is_running", return_value=True)
@patch("everest.detached._query_server", side_effect=query_server_mock)
@patch.object(
    EverestConfig,
    "server_context",
    new_callable=PropertyMock,
    return_value=("localhost", "", ""),
)
@patch("everest.detached.get_opt_status", return_value={})
@patch(
    "everest.bin.monitor_script.everserver_status",
    return_value={"status": ServerStatus.never_run, "message": None},
)
def test_monitor_entry_no_show_all_jobs(
    everserver_status_mock,
    get_opt_status_mock,
    get_server_context_mock,
    query_server_mock,
    server_is_running_mock,
    copy_math_func_test_data_to_tmp,
):
    """Test running everest without --show-all-jobs"""

    # Test when --show-all-jobs flag is not given the shell command are not
    # in the list of forward model jobs
    with capture_streams() as (out, _):
        monitor_entry([CONFIG_FILE_MINIMAL])
    for cmd in all_shell_script_fm_steps:
        assert cmd not in out.getvalue()

    # Check the other jobs are still there
    assert "make_pancakes" in out.getvalue()
    assert "make_scrambled_eggs" in out.getvalue()


@patch(
    "everest.bin.everest_script.run_detached_monitor",
    side_effect=partial(
        run_detached_monitor_mock,
        status=ServerStatus.failed,
        error="Reality was ripped to shreds!",
    ),
)
@patch("everest.bin.everest_script.wait_for_server")
@patch("everest.bin.everest_script.start_server")
def test_exception_raised_when_server_run_fails(
    start_server_mock,
    wait_for_server_mock,
    start_monitor_mock,
    copy_math_func_test_data_to_tmp,
):
    with pytest.raises(SystemExit, match="Reality was ripped to shreds!"):
        everest_entry([CONFIG_FILE_MINIMAL])


@patch("everest.bin.monitor_script.server_is_running", return_value=True)
@patch(
    "everest.bin.monitor_script.run_detached_monitor",
    side_effect=partial(
        run_detached_monitor_mock,
        status=ServerStatus.failed,
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
def test_complete_status_for_normal_run(
    start_server_mock,
    wait_for_server_mock,
    start_monitor_mock,
    copy_math_func_test_data_to_tmp,
):
    everest_entry([CONFIG_FILE_MINIMAL])
    config = EverestConfig.load_file(CONFIG_FILE_MINIMAL)
    status = everserver_status(config)
    expected_status = ServerStatus.completed
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
    status = everserver_status(config)
    expected_status = ServerStatus.completed
    expected_error = None

    assert expected_status == status["status"]
    assert expected_error == status["message"]


@pytest.mark.ui_test
# Application initializes correctly with default arguments
@patch("ieverest.bin.ieverest_script.QApplication")
@patch("ieverest.bin.ieverest_script.IEverest")
@patch("ieverest.bin.ieverest_script.sys.exit")
def test_ieverest_entry(exit_mock, ieverest_mock, q_app_mock):
    ieverest_entry(["config.yml"])

    q_app_mock.assert_called_once_with(["config.yml"])
    app_mock = q_app_mock.return_value
    app_mock.setOrganizationName.assert_called_once_with("Equinor/TNO")
    app_mock.setApplicationName.assert_called_once_with("IEverest")
    app_mock.exec_.assert_called_once()
    ieverest_mock.assert_called_once_with(config_file="config.yml")
    exit_mock.assert_called_once()
