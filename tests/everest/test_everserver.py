import json
import logging
import os
import ssl
from pathlib import Path
from unittest.mock import patch

import pytest

from ert.run_models.everest_run_model import EverestExitCode
from ert.scheduler.event import FinishedEvent
from everest.config import EverestConfig, ServerConfig
from everest.detached import (
    ServerStatus,
    everserver_status,
    start_experiment,
    start_server,
    wait_for_server,
)
from everest.detached.jobs import everserver
from everest.detached.jobs.everserver import ExperimentComplete
from everest.everest_storage import EverestStorage
from everest.simulator import JOB_FAILURE
from everest.strings import (
    SIM_PROGRESS_ENDPOINT,
)


async def wait_for_server_to_complete(config):
    # Wait for the server to complete the optimization.
    # There should be a @pytest.mark.timeout(x) for tests that call this function.
    async def server_running():
        while True:
            event = await driver.event_queue.get()
            if isinstance(event, FinishedEvent) and event.iens == 0:
                return

    driver = await start_server(config, logging.DEBUG)
    try:
        wait_for_server(config.output_dir, 120)
        start_experiment(
            server_context=ServerConfig.get_server_context(config.output_dir),
            config=config,
        )
    except (SystemExit, RuntimeError) as e:
        raise e
    await server_running()


def configure_everserver_logger(*args, **kwargs):
    """Mock exception raised"""
    raise Exception("Configuring logger failed")


def experiment_run(shared_data, server_config, msg_queue):
    msg_queue.put(
        ExperimentComplete(
            exit_code=EverestExitCode.COMPLETED,
            data=shared_data,
        )
    )


def fail_experiment_run(shared_data, server_config, msg_queue):
    shared_data[SIM_PROGRESS_ENDPOINT] = {
        "status": {"failed": 3},
        "progress": [
            [
                {"name": "job1", "status": JOB_FAILURE, "error": "job 1 error 1"},
                {"name": "job1", "status": JOB_FAILURE, "error": "job 1 error 2"},
            ],
            [
                {"name": "job2", "status": JOB_FAILURE, "error": "job 2 error 1"},
            ],
        ],
    }

    msg_queue.put(
        ExperimentComplete(
            msg="Failed",
            exit_code=EverestExitCode.TOO_FEW_REALIZATIONS,
            data=shared_data,
        )
    )


@pytest.mark.integration_test
def test_certificate_generation(change_to_tmpdir):
    cert, key, pw = everserver._generate_certificate(
        ServerConfig.get_certificate_dir("output")
    )

    # check that files are written
    assert os.path.exists(cert)
    assert os.path.exists(key)

    # check certificate is readable
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.load_cert_chain(cert, key, pw)  # raise on error


def test_hostfile_storage(change_to_tmpdir):
    host_file_path = "detach/.session/host_file"

    expected_result = {
        "host": "hostname.1.2.3",
        "port": "5000",
        "cert": "/a/b/c.cert",
        "auth": "1234",
    }
    everserver._write_hostfile(host_file_path, **expected_result)
    assert os.path.exists(host_file_path)
    with open(host_file_path, encoding="utf-8") as f:
        result = json.load(f)
    assert result == expected_result


@patch("sys.argv", ["name", "--output-dir", "everest_output"])
@patch(
    "everest.detached.jobs.everserver._configure_loggers",
    side_effect=configure_everserver_logger,
)
def test_configure_logger_failure(_, change_to_tmpdir):
    everserver.main()
    status = everserver_status(
        ServerConfig.get_everserver_status_path("everest_output")
    )

    assert status["status"] == ServerStatus.failed
    assert "Exception: Configuring logger failed" in status["message"]


@patch("sys.argv", ["name", "--output-dir", "everest_output"])
@patch("everest.detached.jobs.everserver._configure_loggers")
@patch("everest.detached.jobs.everserver._everserver_thread", experiment_run)
def test_status_running_complete(_, change_to_tmpdir):
    everserver.main()

    status = everserver_status(
        ServerConfig.get_everserver_status_path("everest_output")
    )

    assert status["status"] == ServerStatus.completed
    assert status["message"] == "Optimization completed."


@patch("sys.argv", ["name", "--output-dir", "everest_output"])
@patch("everest.detached.jobs.everserver._configure_loggers")
@patch("everest.detached.jobs.everserver._everserver_thread", fail_experiment_run)
def test_status_failed_job(_, change_to_tmpdir):
    everserver.main()

    status = everserver_status(
        ServerConfig.get_everserver_status_path("everest_output")
    )

    # The server should fail and store a user-friendly message.
    assert status["status"] == ServerStatus.failed
    assert "job1 Failed with: job 1 error 1" in status["message"]
    assert "job1 Failed with: job 1 error 2" in status["message"]
    assert "job2 Failed with: job 2 error 1" in status["message"]


@patch("sys.argv", ["name", "--output-dir", "everest_output"])
@patch("everest.detached.jobs.everserver._configure_loggers")
async def test_status_exception(_, change_to_tmpdir, min_config):
    config = EverestConfig(**min_config)

    await wait_for_server_to_complete(config)
    status = everserver_status(
        ServerConfig.get_everserver_status_path("everest_output")
    )

    assert status["status"] == ServerStatus.failed
    assert "Optimization failed:" in status["message"]


@pytest.mark.integration_test
@pytest.mark.xdist_group(name="starts_everest")
@pytest.mark.timeout(240)
@patch("sys.argv", ["name", "--output-dir", "everest_output"])
async def test_status_max_batch_num(copy_math_func_test_data_to_tmp):
    config = EverestConfig.load_file("config_minimal.yml")
    config_dict = {
        **config.model_dump(exclude_none=True),
        "optimization": {"algorithm": "optpp_q_newton", "max_batch_num": 1},
    }
    config = EverestConfig.model_validate(config_dict)

    await wait_for_server_to_complete(config)

    status = everserver_status(
        ServerConfig.get_everserver_status_path(config.output_dir)
    )

    # The server should complete without error.
    assert status["status"] == ServerStatus.completed
    storage = EverestStorage(Path(config.optimization_output_dir))
    storage.read_from_output_dir()

    # Check that there is only one batch.
    assert {b.batch_id for b in storage.data.batches} == {0}


@pytest.mark.integration_test
@pytest.mark.xdist_group(name="starts_everest")
@pytest.mark.timeout(240)
@patch("sys.argv", ["name", "--output-dir", "everest_output"])
async def test_status_contains_max_runtime_failure(change_to_tmpdir, min_config):
    Path("SLEEP_job").write_text("EXECUTABLE sleep", encoding="utf-8")
    min_config["simulator"] = {"max_runtime": 2}
    min_config["forward_model"] = ["sleep 5"]
    min_config["install_jobs"] = [{"name": "sleep", "source": "SLEEP_job"}]

    config = EverestConfig(**min_config)

    await wait_for_server_to_complete(config)

    status = everserver_status(
        ServerConfig.get_everserver_status_path("everest_output")
    )

    assert status["status"] == ServerStatus.failed
    assert (
        "sleep Failed with: The run is cancelled due to reaching MAX_RUNTIME"
        in status["message"]
    )
