import os
from unittest.mock import patch

import pytest
import requests

from ert.config import ErtConfig
from ert.config.queue_config import (
    LocalQueueOptions,
    LsfQueueOptions,
    SlurmQueueOptions,
)
from everest.config import EverestConfig
from everest.config.server_config import ServerConfig
from everest.config.simulator_config import SimulatorConfig
from everest.config_keys import ConfigKeys as CK
from everest.detached import (
    _EVERSERVER_JOB_PATH,
    PROXY,
    ServerStatus,
    _find_res_queue_system,
    everserver_status,
    get_server_queue_options,
    server_is_running,
    start_server,
    stop_server,
    update_everserver_status,
    wait_for_server,
    wait_for_server_to_stop,
)
from everest.simulator.everest_to_ert import _everest_to_ert_config_dict
from everest.strings import (
    DEFAULT_OUTPUT_DIR,
    DETACHED_NODE_DIR,
    EVEREST_SERVER_CONFIG,
    SIMULATION_DIR,
)
from everest.util import makedirs_if_needed


@pytest.mark.flaky(reruns=5)
@pytest.mark.integration_test
@pytest.mark.fails_on_macos_github_workflow
@pytest.mark.xdist_group(name="starts_everest")
async def test_https_requests(copy_math_func_test_data_to_tmp):
    everest_config = EverestConfig.load_file("config_minimal_slow.yml")

    expected_server_status = ServerStatus.never_run
    assert expected_server_status == everserver_status(everest_config)["status"]
    makedirs_if_needed(everest_config.output_dir, roll_if_exists=True)
    await start_server(everest_config)
    try:
        wait_for_server(everest_config, 120)
    except SystemExit as e:
        raise e

    server_status = everserver_status(everest_config)
    assert ServerStatus.running == server_status["status"]

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

    if stop_server(everest_config):
        wait_for_server_to_stop(everest_config, 60)
        server_status = everserver_status(everest_config)

        # Possible the case completed while waiting for the server to stop
        assert server_status["status"] in [
            ServerStatus.stopped,
            ServerStatus.completed,
        ]
        assert not server_is_running(
            *ServerConfig.get_server_context(everest_config.output_dir)
        )
    else:
        server_status = everserver_status(everest_config)
        assert ServerStatus.stopped == server_status["status"]


def test_server_status(copy_math_func_test_data_to_tmp):
    config = EverestConfig.load_file("config_minimal.yml")

    # Check status file does not exist before initial status update
    assert not os.path.exists(
        ServerConfig.get_everserver_status_path(config.output_dir)
    )
    update_everserver_status(config, ServerStatus.starting)

    # Check status file exists after initial status update
    assert os.path.exists(ServerConfig.get_everserver_status_path(config.output_dir))

    # Check we can read the server status from disk
    status = everserver_status(config)
    assert status["status"] == ServerStatus.starting
    assert status["message"] is None

    err_msg_1 = "Danger the universe is preparing for implosion!!!"
    update_everserver_status(config, ServerStatus.failed, message=err_msg_1)
    status = everserver_status(config)
    assert status["status"] == ServerStatus.failed
    assert status["message"] == err_msg_1

    err_msg_2 = "Danger exotic matter detected!!!"
    update_everserver_status(config, ServerStatus.failed, message=err_msg_2)
    status = everserver_status(config)
    assert status["status"] == ServerStatus.failed
    assert status["message"] == "{}\n{}".format(err_msg_1, err_msg_2)

    update_everserver_status(config, ServerStatus.completed)
    status = everserver_status(config)
    assert status["status"] == ServerStatus.completed
    assert status["message"] is not None
    assert status["message"] == "{}\n{}".format(err_msg_1, err_msg_2)


@patch("everest.detached.server_is_running", return_value=False)
def test_wait_for_server(server_is_running_mock, caplog, monkeypatch):
    config = EverestConfig.with_defaults()

    with pytest.raises(RuntimeError, match="Failed to start .* timeout"):
        wait_for_server(config, timeout=1)

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
            "SIMULATION_JOB": [
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
    queue_config = get_server_queue_options(everest_config)

    assert queue_config == LocalQueueOptions(max_running=1)


@pytest.mark.parametrize(
    "queue_system, cores, name",
    [
        ("lsf", 2, None),
        ("slurm", 4, None),
        ("lsf", 3, "test_lsf"),
        ("slurm", 5, "test_slurm"),
    ],
)
def test_everserver_queue_config_equal_to_run_config(
    copy_math_func_test_data_to_tmp, queue_system, cores, name
):
    everest_config, _ = _get_reference_config()

    simulator_config = {CK.QUEUE_SYSTEM: queue_system, CK.CORES: cores}

    if name is not None:
        simulator_config.update({"name": name})
    everest_config.simulator = SimulatorConfig(**simulator_config)
    server_queue_option = get_server_queue_options(everest_config)
    ert_config = _everest_to_ert_config_dict(everest_config)

    run_queue_option = ert_config["QUEUE_OPTION"]

    assert ert_config["QUEUE_SYSTEM"] == server_queue_option.name
    assert (
        next(filter(lambda x: "MAX_RUNNING" in x, reversed(run_queue_option)))[-1]
        == cores
    )
    assert server_queue_option.max_running == 1
    if name is not None:
        option = next(filter(lambda x: name in x, run_queue_option))
        assert option[-1] == name == getattr(server_queue_option, option[1].lower())


@pytest.mark.parametrize("queue_system", ["lsf", "slurm"])
def test_detached_mode_config_only_sim(copy_math_func_test_data_to_tmp, queue_system):
    everest_config, reference = _get_reference_config()

    reference["QUEUE_SYSTEM"] = queue_system.upper()
    queue_options = [(queue_system.upper(), "MAX_RUNNING", 1)]
    reference.setdefault("QUEUE_OPTION", []).extend(queue_options)
    everest_config.simulator = SimulatorConfig(**{CK.QUEUE_SYSTEM: queue_system})
    queue_config = get_server_queue_options(everest_config)
    assert str(queue_config.name.name).lower() == queue_system


def test_detached_mode_config_error(copy_math_func_test_data_to_tmp):
    """
    We are not allowing the simulator queue to be local and at the
    same time the everserver queue to be something other than local
    """
    everest_config, _ = _get_reference_config()

    everest_config.server = ServerConfig(name="server", queue_system="lsf")
    with pytest.raises(ValueError, match="so must the everest server"):
        get_server_queue_options(everest_config)


@pytest.mark.parametrize(
    "config, expected_result",
    [
        (
            EverestConfig.with_defaults(**{CK.SIMULATOR: {CK.QUEUE_SYSTEM: "lsf"}}),
            "LSF",
        ),
        (
            EverestConfig.with_defaults(
                **{
                    CK.SIMULATOR: {CK.QUEUE_SYSTEM: "lsf"},
                    CK.EVERSERVER: {CK.QUEUE_SYSTEM: "lsf"},
                }
            ),
            "LSF",
        ),
        (EverestConfig.with_defaults(**{}), "LOCAL"),
        (
            EverestConfig.with_defaults(**{CK.SIMULATOR: {CK.QUEUE_SYSTEM: "local"}}),
            "LOCAL",
        ),
    ],
)
def test_find_queue_system(config: EverestConfig, expected_result):
    result = _find_res_queue_system(config)

    assert result == expected_result


def test_generate_queue_options_no_config():
    config = EverestConfig.with_defaults(**{})
    assert get_server_queue_options(config) == LocalQueueOptions(max_running=1)


@pytest.mark.parametrize(
    "queue_options, expected_result",
    [
        (
            {"options": "ever_opt_1", "queue_system": "slurm"},
            SlurmQueueOptions(max_running=1),
        ),
        (
            {"options": "ever_opt_1", "queue_system": "lsf"},
            LsfQueueOptions(max_running=1, lsf_resource="ever_opt_1"),
        ),
    ],
)
def test_generate_queue_options_use_simulator_values(queue_options, expected_result):
    config = EverestConfig.with_defaults(**{"simulator": queue_options})
    assert get_server_queue_options(config) == expected_result
