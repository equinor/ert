import logging
import os
import stat
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import requests

import everest
from ert.config import ErtConfig
from ert.config.queue_config import (
    LocalQueueOptions,
    LsfQueueOptions,
    SlurmQueueOptions,
    TorqueQueueOptions,
    activate_script,
)
from ert.scheduler.event import FinishedEvent
from everest.config import EverestConfig, InstallJobConfig
from everest.config.server_config import ServerConfig
from everest.config.simulator_config import SimulatorConfig
from everest.detached import (
    _EVERSERVER_JOB_PATH,
    PROXY,
    ServerStatus,
    everserver_status,
    get_opt_status,
    server_is_running,
    start_server,
    stop_server,
    update_everserver_status,
    wait_for_server,
    wait_for_server_to_stop,
)
from everest.strings import (
    DEFAULT_OUTPUT_DIR,
    DETACHED_NODE_DIR,
    EVEREST_SERVER_CONFIG,
    SIMULATION_DIR,
)
from everest.util import makedirs_if_needed


@pytest.mark.integration_test
@pytest.mark.skip_mac_ci
@pytest.mark.xdist_group(name="starts_everest")
async def test_https_requests(copy_math_func_test_data_to_tmp):
    everest_config = EverestConfig.load_file("config_minimal.yml")
    Path("SLEEP_job").write_text("EXECUTABLE sleep", encoding="utf-8")
    everest_config.forward_model.append("sleep 5")
    everest_config.install_jobs.append(
        InstallJobConfig(name="sleep", source="SLEEP_job")
    )
    # start_server() loads config based on config_path, so we need to actually overwrite it
    everest_config.dump("config_minimal.yml")

    status_path = ServerConfig.get_everserver_status_path(everest_config.output_dir)
    expected_server_status = ServerStatus.never_run
    assert expected_server_status == everserver_status(status_path)["status"]
    makedirs_if_needed(everest_config.output_dir, roll_if_exists=True)
    await start_server(everest_config)
    try:
        wait_for_server(everest_config.output_dir, 240)
    except SystemExit as e:
        raise e

    server_status = everserver_status(status_path)
    assert server_status["status"] in {ServerStatus.running, ServerStatus.starting}

    url, cert, auth = ServerConfig.get_server_context(everest_config.output_dir)
    result = requests.get(url, verify=cert, auth=auth, proxies=PROXY)  # noqa: ASYNC210
    assert result.status_code == 200  # Request has succeeded

    # Test http request fail
    url = url.replace("https", "http")
    with pytest.raises(Exception):  # noqa B017
        response = requests.get(url, verify=cert, auth=auth, proxies=PROXY)  # noqa: ASYNC210
        response.raise_for_status()

    # Test request with wrong password fails
    url, cert, _ = ServerConfig.get_server_context(everest_config.output_dir)
    usr = "admin"
    password = "wrong_password"
    with pytest.raises(Exception):  # noqa B017
        result = requests.get(url, verify=cert, auth=(usr, password), proxies=PROXY)  # noqa: ASYNC210
        result.raise_for_status()

    # Test stopping server
    assert server_is_running(
        *ServerConfig.get_server_context(everest_config.output_dir)
    )
    server_context = ServerConfig.get_server_context(everest_config.output_dir)
    if stop_server(server_context):
        wait_for_server_to_stop(server_context, 240)
        server_status = everserver_status(status_path)

        # Possible the case completed while waiting for the server to stop
        assert server_status["status"] in {ServerStatus.stopped, ServerStatus.completed}
        assert not server_is_running(*server_context)
    else:
        server_status = everserver_status(status_path)
        assert ServerStatus.stopped == server_status["status"]


def test_server_status(copy_math_func_test_data_to_tmp):
    config = EverestConfig.load_file("config_minimal.yml")
    everserver_status_path = ServerConfig.get_everserver_status_path(config.output_dir)
    # Check status file does not exist before initial status update
    assert not os.path.exists(everserver_status_path)
    update_everserver_status(everserver_status_path, ServerStatus.starting)

    # Check status file exists after initial status update
    assert os.path.exists(everserver_status_path)

    # Check we can read the server status from disk
    status = everserver_status(everserver_status_path)
    assert status["status"] == ServerStatus.starting
    assert status["message"] is None

    err_msg_1 = "Danger the universe is preparing for implosion!!!"
    update_everserver_status(
        everserver_status_path, ServerStatus.failed, message=err_msg_1
    )
    status = everserver_status(everserver_status_path)
    assert status["status"] == ServerStatus.failed
    assert status["message"] == err_msg_1

    err_msg_2 = "Danger exotic matter detected!!!"
    update_everserver_status(
        everserver_status_path, ServerStatus.failed, message=err_msg_2
    )
    status = everserver_status(everserver_status_path)
    assert status["status"] == ServerStatus.failed
    assert status["message"] == f"{err_msg_1}\n{err_msg_2}"

    update_everserver_status(everserver_status_path, ServerStatus.completed)
    status = everserver_status(everserver_status_path)
    assert status["status"] == ServerStatus.completed
    assert status["message"] is not None
    assert status["message"] == f"{err_msg_1}\n{err_msg_2}"


@patch("everest.detached.server_is_running", return_value=False)
def test_wait_for_server(server_is_running_mock, caplog):
    config = EverestConfig.with_defaults()

    with pytest.raises(
        RuntimeError, match=r"Failed to get reply from server .* timeout"
    ):
        wait_for_server(config.output_dir, timeout=0.01)

    assert not caplog.messages


def _get_reference_config():
    everest_config = EverestConfig.load_file("config_minimal.yml")
    reference_config = ErtConfig.read_site_config()
    cwd = os.getcwd()
    reference_config.update(
        {
            "INSTALL_JOB": [(EVEREST_SERVER_CONFIG, _EVERSERVER_JOB_PATH)],
            "QUEUE_SYSTEM": "LOCAL",
            "JOBNAME": EVEREST_SERVER_CONFIG,
            "MAX_SUBMIT": 1,
            "NUM_REALIZATIONS": 1,
            "RUNPATH": os.path.join(
                cwd,
                DEFAULT_OUTPUT_DIR,
                DETACHED_NODE_DIR,
                SIMULATION_DIR,
            ),
            "FORWARD_MODEL": [
                [
                    EVEREST_SERVER_CONFIG,
                    "--config-file",
                    os.path.join(cwd, "config_minimal.yml"),
                ],
            ],
            "ENSPATH": os.path.join(
                cwd, DEFAULT_OUTPUT_DIR, DETACHED_NODE_DIR, EVEREST_SERVER_CONFIG
            ),
            "RUNPATH_FILE": os.path.join(
                cwd, DEFAULT_OUTPUT_DIR, DETACHED_NODE_DIR, ".res_runpath_list"
            ),
        }
    )
    return everest_config, reference_config


def test_detached_mode_config_base(copy_math_func_test_data_to_tmp):
    everest_config, _ = _get_reference_config()
    assert everest_config.simulator.queue_system == LocalQueueOptions(max_running=8)


@pytest.mark.parametrize(
    "queue_system, cores",
    [
        ("lsf", 2),
        ("slurm", 4),
        ("lsf", 3),
        ("slurm", 5),
        ("torque", 7),
    ],
)
def test_everserver_queue_config_equal_to_run_config(queue_system, cores):
    simulator_config = {"queue_system": {"name": queue_system, "max_running": cores}}
    everest_config = EverestConfig.with_defaults(simulator=simulator_config)
    everest_config.server.queue_system = SimulatorConfig(**simulator_config)


def test_detached_mode_config_error():
    """
    We are not allowing the simulator queue to be local and at the
    same time the everserver queue to be something other than local
    """
    with pytest.raises(ValueError, match="so must the everest server"):
        EverestConfig.with_defaults(
            simulator={"queue_system": {"name": "local"}},
            server={"queue_system": {"name": "lsf"}},
        )


@pytest.mark.parametrize(
    "config, expected_result",
    [
        (
            EverestConfig.with_defaults(simulator={"queue_system": {"name": "lsf"}}),
            "lsf",
        ),
        (
            EverestConfig.with_defaults(
                simulator={"queue_system": {"name": "lsf"}},
                server={"queue_system": {"name": "lsf"}},
            ),
            "lsf",
        ),
        (EverestConfig.with_defaults(), "local"),
        (
            EverestConfig.with_defaults(simulator={"queue_system": {"name": "local"}}),
            "local",
        ),
    ],
)
def test_find_queue_system(config: EverestConfig, expected_result):
    result = config.simulator

    assert result.queue_system.name == expected_result


def test_generate_queue_options_no_config():
    config = EverestConfig.with_defaults()
    assert config.server.queue_system == LocalQueueOptions(max_running=1)


@pytest.mark.parametrize(
    "queue_options, expected_result",
    [
        (
            {"partition": "ever_opt_1", "name": "slurm"},
            SlurmQueueOptions(max_running=1, partition="ever_opt_1"),
        ),
        (
            {"lsf_queue": "ever_opt_1", "name": "lsf"},
            LsfQueueOptions(
                max_running=1,
                lsf_queue="ever_opt_1",
            ),
        ),
        (
            {
                "name": "torque",
                "keep_qsub_output": "1",
            },
            TorqueQueueOptions(max_running=1, keep_qsub_output=True),
        ),
    ],
)
def test_generate_queue_options_use_simulator_values(
    queue_options, expected_result, monkeypatch
):
    monkeypatch.setattr(
        everest.config.server_config.ErtPluginManager,
        "activate_script",
        MagicMock(return_value=activate_script()),
    )
    config = EverestConfig.with_defaults(simulator={"queue_system": queue_options})
    assert config.server.queue_system == expected_result


@pytest.mark.timeout(5)  # Simulation might not finish
@pytest.mark.integration_test
@pytest.mark.xdist_group(name="starts_everest")
async def test_starting_not_in_folder(tmp_path, monkeypatch):
    """
    This tests that the second argument to the everserver is the config
    file, and that the config file exists. This is a regression test for
    a bug that happened when everest was started from a different dir
    than the config file was in.
    """

    async def server_running():
        while True:
            event = await driver.event_queue.get()
            if isinstance(event, FinishedEvent) and event.iens == 0:
                return event

    os.makedirs(tmp_path / "new_folder")
    monkeypatch.chdir(tmp_path / "new_folder")
    everest_config = EverestConfig.with_defaults()
    everest_config.dump("minimal_config.yml")
    config_dict = {
        **everest_config.model_dump(exclude_none=True),
        "config_path": str(Path("minimal_config.yml").absolute()),
    }
    everest_config = EverestConfig.model_validate(config_dict)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", f".:{os.environ['PATH']}")
    everserver_path = Path("everserver")
    with open(everserver_path, "w", encoding="utf-8") as file:  # noqa: ASYNC230
        file.write(
            """#!/usr/bin/env python
import sys
from pathlib import Path
if __name__ == "__main__":
    config_path = sys.argv[2]
    if not Path(config_path).exists():
        raise ValueError(f"config_path ({config_path}) does not exist")
"""
        )
    everserver_path.chmod(everserver_path.stat().st_mode | stat.S_IEXEC)
    makedirs_if_needed(everest_config.output_dir, roll_if_exists=True)
    driver = await start_server(everest_config, logging_level=logging.DEBUG)
    final_state = await server_running()
    assert final_state.returncode == 0


def test_get_opt_status(cached_example):
    _, config_file, _, _ = cached_example("math_func/config_multiobj.yml")
    config = EverestConfig.load_file(config_file)

    opts = get_opt_status(config.optimization_output_dir)

    assert np.allclose(
        opts["objective_history"], [-2.3333, -2.3335, -2.0000], atol=1e-4
    )

    assert np.allclose(
        opts["control_history"]["point.x"],
        [0.0, -0.004202181916184627, -0.0021007888698514315],
        atol=1e-4,
    )
    assert np.allclose(
        opts["control_history"]["point.y"],
        [0.0, -0.011298196942730383, -0.0056482862617779715],
        atol=1e-4,
    )
    assert np.allclose(
        opts["control_history"]["point.z"], [0.0, 1.0, 0.4999281115746754], atol=1e-4
    )

    assert np.allclose(
        opts["objectives_history"]["distance_p"],
        [-0.75, -0.7656459808349609, -0.5077850222587585],
        atol=1e-4,
    )
    assert np.allclose(
        opts["objectives_history"]["distance_q"],
        [-4.75, -4.703639984130859, -4.476789951324463],
        atol=1e-4,
    )

    assert opts["accepted_control_indices"] == [0, 2]

    cmond = opts["cli_monitor_data"]

    assert cmond["batches"] == [0, 1, 2]
    assert cmond["controls"][0]["point.x"] == 0.0
    assert cmond["controls"][0]["point.y"] == 0.0
    assert cmond["controls"][0]["point.z"] == 0.0

    assert np.allclose(
        cmond["controls"][1]["point.x"], -0.004202181916184627, atol=1e-4
    )
    assert np.allclose(
        cmond["controls"][1]["point.y"], -0.011298196942730383, atol=1e-4
    )
    assert np.allclose(cmond["controls"][1]["point.z"], 1.0, atol=1e-4)
    assert np.allclose(
        cmond["controls"][2]["point.x"], -0.0021007888698514315, atol=1e-4
    )
    assert np.allclose(
        cmond["controls"][2]["point.y"], -0.0056482862617779715, atol=1e-4
    )
    assert np.allclose(cmond["controls"][2]["point.z"], 0.4999281115746754, atol=1e-4)

    assert np.allclose(
        cmond["objective_value"],
        [-2.333333333333333, -2.333525975545247, -2.000048339366913],
        atol=1e-4,
    )
    assert np.allclose(
        cmond["expected_objectives"]["distance_p"],
        [-0.75, -0.7656459808349609, -0.5077850222587585],
        atol=1e-4,
    )
    assert np.allclose(
        cmond["expected_objectives"]["distance_q"],
        [-4.75, -4.703639984130859, -4.476789951324463],
        atol=1e-4,
    )
