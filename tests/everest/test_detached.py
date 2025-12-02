import logging
import os
import stat
from functools import partial
from pathlib import Path
from shutil import which
from unittest import mock
from unittest.mock import MagicMock, patch

import pytest
import requests
import yaml

import ert
from ert import plugin
from ert.config.queue_config import (
    LocalQueueOptions,
    LsfQueueOptions,
    SlurmQueueOptions,
    TorqueQueueOptions,
    activate_script,
)
from ert.dark_storage.client import ErtClientConnectionInfo
from ert.plugins import ErtRuntimePlugins
from ert.scheduler.event import FinishedEvent
from ert.services import ErtServer
from everest.config import EverestConfig
from everest.config.forward_model_config import ForwardModelStepConfig
from everest.config.install_job_config import InstallForwardModelStepConfig
from everest.config.server_config import ServerConfig
from everest.config.simulator_config import SimulatorConfig
from everest.detached import (
    PROXY,
    server_is_running,
    start_server,
    stop_server,
    wait_for_server,
    wait_for_server_to_stop,
)
from everest.util import makedirs_if_needed
from tests.everest.utils import everest_config_with_defaults


@pytest.mark.integration_test
@pytest.mark.skip_mac_ci
@pytest.mark.xdist_group(name="starts_everest")
@pytest.mark.usefixtures("use_site_configurations_with_no_queue_options")
async def test_https_requests(change_to_tmpdir):
    Path("./config.yml").touch()
    everest_config = everest_config_with_defaults(config_path="./config.yml")
    everest_config.forward_model.append(ForwardModelStepConfig(job="sleep 5"))
    everest_config.install_jobs.append(
        InstallForwardModelStepConfig(name="sleep", executable=f"{which('sleep')}")
    )
    # start_server() loads config based on config_path, so we need to actually
    # overwrite it
    everest_config.dump("config.yml")

    makedirs_if_needed(everest_config.output_dir, roll_if_exists=True)
    await start_server(everest_config, logging_level=logging.INFO)

    session = ErtServer.session(
        Path(ServerConfig.get_session_dir(everest_config.output_dir)), 240
    )
    wait_for_server(session, 240)
    url, cert, auth = ServerConfig.get_server_context_from_conn_info(session.conn_info)
    result = requests.get(url, verify=cert, auth=auth, proxies=PROXY)  # noqa: ASYNC210
    assert result.status_code == 200  # Request has succeeded

    # Test http request fail
    http_url = url.replace("https", "http")
    with pytest.raises(Exception):  # noqa B017
        response = requests.get(http_url, verify=cert, auth=auth, proxies=PROXY)  # noqa: ASYNC210
        response.raise_for_status()

    # Test request with wrong password fails
    auth = ("admin", "wrong_password")
    result = requests.get(url, verify=cert, auth=auth, proxies=PROXY)  # noqa: ASYNC210

    assert result.status_code == 401  # Unauthorized

    # Test stopping server
    assert server_is_running(
        *ServerConfig.get_server_context_from_conn_info(session.conn_info)
    )
    server_context = ServerConfig.get_server_context_from_conn_info(session.conn_info)
    if stop_server(server_context):
        wait_for_server_to_stop(server_context, 240)
        assert not server_is_running(*server_context)


@patch("everest.detached.server_is_running", return_value=False)
@patch(
    "everest.config.ServerConfig.get_server_context_from_conn_info",
    return_value=("url", "cert", ("user", "token")),
)
def test_wait_for_server(mock_get_context, mock_is_running):
    client = MagicMock()
    with pytest.raises(
        RuntimeError, match=r"Failed to get reply from server within .* seconds"
    ):
        wait_for_server(client, timeout=0.01)


@pytest.mark.usefixtures("use_site_configurations_with_no_queue_options")
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
    everest_config = everest_config_with_defaults(simulator=simulator_config)
    everest_config.server.queue_system = SimulatorConfig(**simulator_config)


def test_detached_mode_config_error():
    """
    We are not allowing the simulator queue to be local and at the
    same time the everserver queue to be something other than local
    """
    with pytest.raises(ValueError, match="so must the everest server"):
        everest_config_with_defaults(
            simulator={"queue_system": {"name": "local"}},
            server={"queue_system": {"name": "lsf"}},
        )


@pytest.mark.usefixtures("use_site_configurations_with_no_queue_options")
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
    config = everest_config_with_defaults(**config_kwargs)

    result = config.simulator
    assert result.queue_system.name == expected_result


@pytest.mark.usefixtures("use_site_configurations_with_no_queue_options")
def test_generate_queue_options_no_config():
    config = everest_config_with_defaults()
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
@pytest.mark.usefixtures("use_site_configurations_with_no_queue_options")
def test_that_server_queue_system_defaults_to_simulator_queue_options(
    queue_class, expected_queue_kwargs
):
    config = everest_config_with_defaults(
        simulator={"queue_system": expected_queue_kwargs}
    )
    expected_result = queue_class(**expected_queue_kwargs)
    assert config.server.queue_system == expected_result


@pytest.mark.usefixtures("use_site_configurations_with_no_queue_options")
@pytest.mark.parametrize("use_plugin", (False,))
@pytest.mark.parametrize(
    "queue_options",
    [
        {"name": "slurm", "activate_script": "From user"},
        {"name": "slurm"},
    ],
)
def test_queue_options_site_config(queue_options, use_plugin, min_config):
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
        ert.plugins.plugin_manager.ErtPluginManager, plugins=plugins
    )
    with (
        patch("ert.plugins.ErtPluginManager", patched_everest),
    ):
        config = EverestConfig.with_plugins(
            {"simulator": {"queue_system": queue_options}} | min_config
        )
    assert config.simulator.queue_system.activate_script == expected_result


@pytest.mark.usefixtures("use_site_configurations_with_no_queue_options")
@pytest.mark.parametrize("use_plugin", (True, False))
@pytest.mark.parametrize(
    "queue_options",
    [
        {"queue_system": {"name": "slurm"}},
        {},
    ],
)
def test_simulator_queue_system_site_config(queue_options, use_plugin, min_config):
    if queue_options:
        expected_result = SlurmQueueOptions  # User specified
    elif use_plugin:
        expected_result = LsfQueueOptions  # Mock site config
    else:
        expected_result = LocalQueueOptions  # Default value

    if use_plugin:
        runtime_plugins_with_lsfqueue = ErtRuntimePlugins(
            queue_options=LsfQueueOptions()
        )
        with mock.patch(
            "ert.plugins.plugin_manager.ErtRuntimePlugins",
            return_value=runtime_plugins_with_lsfqueue,
        ):
            config = EverestConfig.with_plugins(
                {"simulator": queue_options} | min_config
            )
    else:
        config = EverestConfig.model_validate({"simulator": queue_options} | min_config)

    assert isinstance(config.simulator.queue_system, expected_result)


@pytest.mark.timeout(5)  # Simulation might not finish
@pytest.mark.integration_test
@pytest.mark.xdist_group(name="starts_everest")
@pytest.mark.usefixtures("use_site_configurations_with_no_queue_options")
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
    everest_config = everest_config_with_defaults()
    everest_config.dump("minimal_config.yml")
    config_dict = {
        **everest_config.model_dump(exclude_none=True),
        "config_path": str(Path("minimal_config.yml").absolute()),
    }
    everest_config = EverestConfig.model_validate(config_dict)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", f".:{os.environ['PATH']}")
    everserver_path = Path("everserver")
    everserver_path.write_text(
        """#!/usr/bin/env python
import sys
from pathlib import Path
if __name__ == "__main__":
    config_path = sys.argv[2]
    if not Path(config_path).exists():
        raise ValueError(f"config_path ({config_path}) does not exist")
""",
        encoding="utf-8",
    )
    everserver_path.chmod(everserver_path.stat().st_mode | stat.S_IEXEC)
    makedirs_if_needed(everest_config.output_dir, roll_if_exists=True)
    driver = await start_server(everest_config, logging_level=logging.DEBUG)
    final_state = await server_running()
    assert final_state.returncode == 0


def test_get_that_get_server_info_from_conn_info_converts_values():
    conn_info = ErtClientConnectionInfo(
        base_url="https://example.com:1234",
        cert="/path/to/cert.pem",
        auth_token="sometoken",
    )
    url, cert_file, auth = ServerConfig.get_server_context_from_conn_info(conn_info)
    assert url == "https://example.com:1234/experiment_server"
    assert cert_file == "/path/to/cert.pem"
    assert auth == ("username", "sometoken")


@pytest.mark.parametrize(
    "conn_info, expected_exception, expected_message",
    [
        (
            ErtClientConnectionInfo(
                base_url="https://example.com:1234",
                cert="/path/to/cert.pem",
                auth_token=None,
            ),
            RuntimeError,
            "No authentication token found in storage session",
        ),
        (
            ErtClientConnectionInfo(
                base_url="https://example.com:1234",
                cert=False,
                auth_token="sometoken",
            ),
            RuntimeError,
            "Invalid certificate file in storage session",
        ),
        (
            ErtClientConnectionInfo(
                base_url="https://example.com:1234",
                cert=True,
                auth_token="sometoken",
            ),
            RuntimeError,
            "Invalid certificate file in storage session",
        ),
    ],
)
def test_that_get_server_context_from_conn_info_raises_on_wrong_input(
    conn_info, expected_exception, expected_message
):
    with pytest.raises(expected_exception) as exc_info:
        ServerConfig.get_server_context_from_conn_info(conn_info)
    assert str(exc_info.value) == expected_message
