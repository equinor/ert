import logging
import os
from collections import namedtuple
from unittest.mock import patch

import pytest
import requests

from ert.config import ErtConfig, QueueSystem
from ert.storage import open_storage
from everest.config import EverestConfig
from everest.config.server_config import ServerConfig
from everest.config.simulator_config import SimulatorConfig
from everest.config_keys import ConfigKeys as CK
from everest.detached import (
    _EVERSERVER_JOB_PATH,
    PROXY,
    ServerStatus,
    _find_res_queue_system,
    _generate_queue_options,
    context_stop_and_wait,
    everserver_status,
    generate_everserver_ert_config,
    server_is_running,
    start_server,
    stop_server,
    update_everserver_status,
    wait_for_context,
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


class MockContext:
    def __init__(self):
        pass

    @staticmethod
    def has_job_failed(*args):
        return True

    @staticmethod
    def job_progress(*args):
        job = namedtuple("Job", "std_err_file")
        job.std_err_file = "error_file.0"
        job_progress = namedtuple("JobProgres", ["jobs"])
        job_progress.steps = [job]
        return job_progress


@pytest.mark.flaky(reruns=5)
@pytest.mark.integration_test
@pytest.mark.fails_on_macos_github_workflow
@pytest.mark.xdist_group(name="starts_everest")
def test_https_requests(copy_math_func_test_data_to_tmp):
    everest_config = EverestConfig.load_file("config_minimal_slow.yml")

    expected_server_status = ServerStatus.never_run
    assert expected_server_status == everserver_status(everest_config)["status"]
    wait_for_context()
    ert_config = ErtConfig.with_plugins().from_dict(
        generate_everserver_ert_config(everest_config)
    )
    makedirs_if_needed(everest_config.output_dir, roll_if_exists=True)
    with open_storage(ert_config.ens_path, "w") as storage:
        start_server(everest_config, ert_config, storage)
        try:
            wait_for_server(everest_config, 120)
        except SystemExit as e:
            context_stop_and_wait()
            raise e

        server_status = everserver_status(everest_config)
        assert ServerStatus.running == server_status["status"]

        url, cert, auth = everest_config.server_context
        result = requests.get(url, verify=cert, auth=auth, proxies=PROXY)
        assert result.status_code == 200  # Request has succeeded

        # Test http request fail
        url = url.replace("https", "http")
        with pytest.raises(Exception):  # noqa B017
            response = requests.get(url, verify=cert, auth=auth, proxies=PROXY)
            response.raise_for_status()

        # Test request with wrong password fails
        url, cert, _ = everest_config.server_context
        usr = "admin"
        password = "wrong_password"
        with pytest.raises(Exception):  # noqa B017
            result = requests.get(url, verify=cert, auth=(usr, password), proxies=PROXY)
            result.raise_for_status()

        # Test stopping server
        assert server_is_running(everest_config)

        if stop_server(everest_config):
            wait_for_server_to_stop(everest_config, 60)
            context_stop_and_wait()
            server_status = everserver_status(everest_config)

            # Possible the case completed while waiting for the server to stop
            assert server_status["status"] in [
                ServerStatus.stopped,
                ServerStatus.completed,
            ]
            assert not server_is_running(everest_config)
        else:
            context_stop_and_wait()
            server_status = everserver_status(everest_config)
            assert ServerStatus.stopped == server_status["status"]


def test_server_status(copy_math_func_test_data_to_tmp):
    config = EverestConfig.load_file("config_minimal.yml")

    # Check status file does not exist before initial status update
    assert not os.path.exists(config.everserver_status_path)
    update_everserver_status(config, ServerStatus.starting)

    # Check status file exists after initial status update
    assert os.path.exists(config.everserver_status_path)

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
def test_wait_for_server(
    server_is_running_mock, caplog, copy_test_data_to_tmp, monkeypatch
):
    monkeypatch.chdir("detached")
    config = EverestConfig.load_file("valid_yaml_config.yml")

    with caplog.at_level(logging.DEBUG), pytest.raises(RuntimeError):
        wait_for_server(config, timeout=1, context=None)

    assert not caplog.messages
    context = MockContext()
    with caplog.at_level(logging.DEBUG), pytest.raises(SystemExit):
        wait_for_server(config, timeout=120, context=context)

    expected_error_msg = (
        'Error when parsing config_file:"DISTANCE3" '
        "Keyword:ARGLIST must have at least 1 arguments.\n"
        "Error message: ext_joblist_get_job_copy: "
        "asked for job:distance3 which does not exist\n"
        "Error message: Program received signal:6"
    )

    assert expected_error_msg in "\n".join(caplog.messages)

    server_status = everserver_status(config)
    assert server_status["status"] == ServerStatus.failed
    assert server_status["message"] == expected_error_msg


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
    everest_config, reference = _get_reference_config()
    ert_config = generate_everserver_ert_config(everest_config)

    assert ert_config is not None
    assert ert_config == reference


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
    server_ert_config = generate_everserver_ert_config(everest_config)
    ert_config = _everest_to_ert_config_dict(everest_config)

    server_queue_option = server_ert_config["QUEUE_OPTION"]
    run_queue_option = ert_config["QUEUE_OPTION"]

    assert ert_config["QUEUE_SYSTEM"] == server_ert_config["QUEUE_SYSTEM"]
    assert (
        next(filter(lambda x: "MAX_RUNNING" in x, reversed(run_queue_option)))[-1]
        == cores
    )
    assert (
        next(filter(lambda x: "MAX_RUNNING" in x, reversed(server_queue_option)))[-1]
        == 1
    )
    if name is not None:
        assert next(filter(lambda x: name in x, run_queue_option)) == next(
            filter(lambda x: name in x, server_queue_option)
        )


def test_detached_mode_config_debug(copy_math_func_test_data_to_tmp):
    everest_config, reference = _get_reference_config()
    ert_config = generate_everserver_ert_config(everest_config, debug_mode=True)

    reference["SIMULATION_JOB"][0].append("--debug")

    assert ert_config is not None
    assert ert_config == reference


@pytest.mark.parametrize("queue_system", ["lsf", "slurm"])
def test_detached_mode_config_only_sim(copy_math_func_test_data_to_tmp, queue_system):
    everest_config, reference = _get_reference_config()

    reference["QUEUE_SYSTEM"] = queue_system.upper()
    queue_options = [(queue_system.upper(), "MAX_RUNNING", 1)]
    reference.setdefault("QUEUE_OPTION", []).extend(queue_options)
    everest_config.simulator = SimulatorConfig(**{CK.QUEUE_SYSTEM: queue_system})
    ert_config = generate_everserver_ert_config(everest_config)
    assert ert_config is not None
    assert ert_config == reference


def test_detached_mode_config_error(copy_math_func_test_data_to_tmp):
    """
    We are not allowing the simulator queue to be local and at the
    same time the everserver queue to be something other than local
    """
    everest_config, _ = _get_reference_config()

    everest_config.server = ServerConfig(name="server", queue_system="lsf")
    with pytest.raises(ValueError):
        generate_everserver_ert_config(everest_config)


def test_detached_mode_config_queue_name(copy_math_func_test_data_to_tmp):
    everest_config, reference = _get_reference_config()

    queue_name = "put_me_in_the_queue"
    reference["QUEUE_SYSTEM"] = QueueSystem.LSF
    queue_options = [(QueueSystem.LSF, "LSF_QUEUE", queue_name)]

    reference.setdefault("QUEUE_OPTION", []).extend(queue_options)
    everest_config.simulator = SimulatorConfig(queue_system="lsf")
    everest_config.server = ServerConfig(queue_system="lsf", name=queue_name)

    ert_config = generate_everserver_ert_config(everest_config)
    assert ert_config is not None
    assert ert_config == reference


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


def test_find_queue_system_error():
    config = EverestConfig.with_defaults(**{"server": {CK.QUEUE_SYSTEM: "lsf"}})

    with pytest.raises(ValueError):
        _find_res_queue_system(config)


@pytest.mark.parametrize("queue_options", [[], [("EVEREST_KEY", "RES_KEY")]])
@pytest.mark.parametrize("queue_system", ["LSF", "SLURM", "SOME_NEW_QUEUE_SYSTEM"])
def test_generate_queue_options_no_config(queue_options, queue_system):
    config = EverestConfig.with_defaults(**{})
    res_queue_name = "SOME_ERT_KEY"  # LSF_QUEUE_KEY for LSF
    assert [(queue_system, "MAX_RUNNING", 1)] == _generate_queue_options(
        config, queue_options, res_queue_name, queue_system
    )


@pytest.mark.parametrize("queue_options", [[], [("exclude_host", "RES_KEY")]])
@pytest.mark.parametrize("queue_system", ["LSF", "SLURM", "SOME_NEW_QUEUE_SYSTEM"])
def test_generate_queue_options_only_name(queue_options, queue_system):
    config = EverestConfig.with_defaults(**{"server": {"name": "my_custom_queue_name"}})
    res_queue_name = "SOME_ERT_KEY"  # LSF_QUEUE_KEY for LSF
    assert _generate_queue_options(
        config, queue_options, res_queue_name, queue_system
    ) == [
        (
            queue_system,
            res_queue_name,
            "my_custom_queue_name",
        ),
    ]


@pytest.mark.parametrize(
    "queue_options, expected_result",
    [
        ([], []),
        (
            [("options", "RES_KEY")],
            [
                (
                    "SOME_QUEUE_SYSTEM",
                    "RES_KEY",
                    "ever_opt_1",
                ),
            ],
        ),
    ],
)
def test_generate_queue_options_only_options(queue_options, expected_result):
    config = EverestConfig.with_defaults(**{"server": {"options": "ever_opt_1"}})
    res_queue_name = "NOT_RELEVANT_IN_THIS_CONTEXT"
    queue_system = "SOME_QUEUE_SYSTEM"
    assert (
        _generate_queue_options(config, queue_options, res_queue_name, queue_system)
        == expected_result
    )


@pytest.mark.parametrize(
    "queue_options, expected_result",
    [
        (
            [],
            [
                (
                    "SLURM",
                    "MAX_RUNNING",
                    1,
                )
            ],
        ),
        (
            [("options", "RES_KEY")],
            [
                (
                    "SLURM",
                    "MAX_RUNNING",
                    1,
                ),
                (
                    "SLURM",
                    "RES_KEY",
                    "ever_opt_1",
                ),
            ],
        ),
    ],
)
def test_generate_queue_options_use_simulator_values(queue_options, expected_result):
    config = EverestConfig.with_defaults(**{"simulator": {"options": "ever_opt_1"}})
    res_queue_name = "NOT_RELEVANT_IN_THIS_CONTEXT"
    queue_system = "SLURM"
    assert (
        _generate_queue_options(config, queue_options, res_queue_name, queue_system)
        == expected_result
    )
