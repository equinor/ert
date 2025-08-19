import logging
import os
import stat
from functools import partial
from pathlib import Path
from shutil import which
from unittest.mock import MagicMock, patch

import pytest
import requests
import yaml

import everest
from ert import plugin
from ert.config import QueueSystem
from ert.config.queue_config import (
    LocalQueueOptions,
    LsfQueueOptions,
    SlurmQueueOptions,
    TorqueQueueOptions,
    activate_script,
)
from ert.scheduler.event import FinishedEvent
from everest.config import EverestConfig, InstallJobConfig
from everest.config.forward_model_config import ForwardModelStepConfig
from everest.config.server_config import ServerConfig
from everest.config.simulator_config import SimulatorConfig
from everest.detached import (
    PROXY,
    ExperimentState,
    everserver_status,
    server_is_running,
    start_server,
    stop_server,
    update_everserver_status,
    wait_for_server,
    wait_for_server_to_stop,
)
from everest.util import makedirs_if_needed


@pytest.mark.integration_test
@pytest.mark.skip_mac_ci
@pytest.mark.xdist_group(name="starts_everest")
async def test_https_requests(change_to_tmpdir):
    Path("./config.yml").touch()
    everest_config = EverestConfig.with_defaults(config_path="./config.yml")
    everest_config.forward_model.append(ForwardModelStepConfig(job="sleep 5"))
    everest_config.install_jobs.append(
        InstallJobConfig(name="sleep", executable=f"{which('sleep')}")
    )
    # start_server() loads config based on config_path, so we need to actually
    # overwrite it
    everest_config.dump("config.yml")

    status_path = ServerConfig.get_everserver_status_path(everest_config.output_dir)
    expected_server_status = ExperimentState.never_run
    assert expected_server_status == everserver_status(status_path)["status"]
    makedirs_if_needed(everest_config.output_dir, roll_if_exists=True)
    await start_server(everest_config, logging_level=logging.INFO)

    wait_for_server(everest_config.output_dir, 240)

    server_status = everserver_status(status_path)
    assert server_status["status"] in {
        ExperimentState.running,
        ExperimentState.pending,
    }

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
        assert server_status["status"] in {
            ExperimentState.stopped,
            ExperimentState.completed,
        }
        assert not server_is_running(*server_context)
    else:
        server_status = everserver_status(status_path)
        assert ExperimentState.stopped == server_status["status"]


def test_server_status(change_to_tmpdir):
    config = EverestConfig.with_defaults()

    everserver_status_path = ServerConfig.get_everserver_status_path(config.output_dir)
    # Check status file does not exist before initial status update
    assert not os.path.exists(everserver_status_path)
    update_everserver_status(everserver_status_path, ExperimentState.pending)

    # Check status file exists after initial status update
    assert os.path.exists(everserver_status_path)

    # Check we can read the server status from disk
    status = everserver_status(everserver_status_path)
    assert status["status"] == ExperimentState.pending
    assert status["message"] is None

    err_msg_1 = "Danger the universe is preparing for implosion!!!"
    update_everserver_status(
        everserver_status_path, ExperimentState.failed, message=err_msg_1
    )
    status = everserver_status(everserver_status_path)
    assert status["status"] == ExperimentState.failed
    assert status["message"] == err_msg_1

    err_msg_2 = "Danger exotic matter detected!!!"
    update_everserver_status(
        everserver_status_path, ExperimentState.failed, message=err_msg_2
    )
    status = everserver_status(everserver_status_path)
    assert status["status"] == ExperimentState.failed
    assert status["message"] == f"{err_msg_1}\n{err_msg_2}"

    update_everserver_status(everserver_status_path, ExperimentState.completed)
    status = everserver_status(everserver_status_path)
    assert status["status"] == ExperimentState.completed
    assert status["message"] is not None
    assert status["message"] == f"{err_msg_1}\n{err_msg_2}"


@patch("everest.detached.server_is_running", return_value=False)
def test_wait_for_server(_):
    config = EverestConfig.with_defaults()

    with pytest.raises(
        RuntimeError, match=r"Failed to get reply from server within .* seconds"
    ):
        wait_for_server(config.output_dir, timeout=0.01)


@pytest.mark.usefixtures("no_plugins")
def test_detached_mode_config_base(min_config, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    with open("config.yml", "w", encoding="utf-8") as fout:
        yaml.dump(min_config, fout)
    everest_config = EverestConfig.load_file("config.yml")

    # Expect it to take the default of the ERT QueueConfig
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
    "config_kwargs, expected_result",
    [
        ({"simulator": {"queue_system": {"name": "lsf"}}}, "lsf"),
        (
            {
                "simulator": {"queue_system": {"name": "lsf"}},
                "server": {"queue_system": {"name": "lsf"}},
            },
            "lsf",
        ),
        ({}, "local"),
        ({"simulator": {"queue_system": {"name": "local"}}}, "local"),
    ],
)
def test_find_queue_system(config_kwargs, expected_result):
    config = EverestConfig.with_defaults(**config_kwargs)

    result = config.simulator
    assert result.queue_system.name == expected_result


@pytest.mark.usefixtures("no_plugins")
def test_generate_queue_options_no_config():
    config = EverestConfig.with_defaults()
    assert config.server.queue_system == LocalQueueOptions(max_running=1)


@pytest.mark.parametrize(
    "queue_class, expected_queue_kwargs",
    [
        (
            SlurmQueueOptions,
            {"name": "slurm", "partition": "ever_opt_1", "max_running": 1},
        ),
        (LsfQueueOptions, {"name": "lsf", "lsf_queue": "ever_opt_1", "max_running": 1}),
        (
            TorqueQueueOptions,
            {"name": "torque", "keep_qsub_output": True, "max_running": 1},
        ),
    ],
)
def test_that_server_queue_system_defaults_to_simulator_queue_options(
    monkeypatch, queue_class, expected_queue_kwargs
):
    monkeypatch.setattr(
        everest.config.everest_config.ErtPluginManager,
        "activate_script",
        MagicMock(return_value=activate_script()),
    )

    config = EverestConfig.with_defaults(
        simulator={"queue_system": expected_queue_kwargs}
    )
    expected_result = queue_class(**expected_queue_kwargs)
    assert config.server.queue_system == expected_result


@pytest.mark.parametrize("use_plugin", (True, False))
@pytest.mark.parametrize(
    "queue_options",
    [
        {"name": "slurm", "activate_script": "From user"},
        {"name": "slurm"},
    ],
)
def test_queue_options_site_config(queue_options, use_plugin, monkeypatch, min_config):
    plugin_result = "From plugin"
    if "activate_script" in queue_options:
        expected_result = queue_options["activate_script"]
    elif use_plugin:
        expected_result = plugin_result
    else:
        expected_result = activate_script()

    plugins = []
    if use_plugin:

        class ActivatePlugin:
            @plugin(name="first")
            def activate_script(self):
                return plugin_result

        plugins = [ActivatePlugin()]
    patched_everest = partial(
        everest.config.everest_config.ErtPluginManager, plugins=plugins
    )
    with (
        patch("everest.config.everest_config.ErtPluginManager", patched_everest),
    ):
        config = EverestConfig.with_plugins(
            {"simulator": {"queue_system": queue_options}} | min_config
        )
    assert config.simulator.queue_system.activate_script == expected_result


@pytest.mark.parametrize("use_plugin", (True, False))
@pytest.mark.parametrize(
    "queue_options",
    [
        {"queue_system": {"name": "slurm"}},
        {},
    ],
)
def test_simulator_queue_system_site_config(
    queue_options, use_plugin, monkeypatch, min_config
):
    if queue_options:
        expected_result = SlurmQueueOptions  # User specified
    elif use_plugin:
        expected_result = LsfQueueOptions  # Mock site config
    else:
        expected_result = LocalQueueOptions  # Default value
    if use_plugin:
        monkeypatch.setattr(
            everest.config.everest_config.ErtConfig,
            "read_site_config",
            MagicMock(return_value={"QUEUE_SYSTEM": QueueSystem.LSF}),
        )
    config = EverestConfig.with_plugins({"simulator": queue_options} | min_config)
    assert isinstance(config.simulator.queue_system, expected_result)


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
