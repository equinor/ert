import json
import os
import ssl
from pathlib import Path
from unittest.mock import patch

import pytest
import requests
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

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
from everest.everest_storage import EverestStorage
from everest.simulator import JOB_FAILURE, JOB_SUCCESS
from everest.strings import (
    OPT_FAILURE_REALIZATIONS,
    SIM_PROGRESS_ENDPOINT,
    STOP_ENDPOINT,
)


async def wait_for_server_to_complete(config):
    # Wait for the server to complete the optimization.
    # There should be a @pytest.mark.timeout(x) for tests that call this function.
    async def server_running():
        while True:
            event = await driver.event_queue.get()
            if isinstance(event, FinishedEvent) and event.iens == 0:
                return

    driver = await start_server(config, debug=True)
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


def check_status(*args, **kwargs):
    everest_server_status_path = str(Path(args[0]).parent / "status")
    status = everserver_status(everest_server_status_path)
    assert status["status"] == kwargs["status"]


def fail_optimization(self, from_ropt=False):
    # Patch start_optimization to raise a failed optimization callback. Also
    # call the provided simulation callback, which has access to the shared_data
    # variable in the eversever main function. Patch that callback to modify
    # shared_data (see set_shared_status() below).
    self._sim_callback(None)
    if from_ropt:
        self._exit_code = EverestExitCode.TOO_FEW_REALIZATIONS
        return EverestExitCode.TOO_FEW_REALIZATIONS

    raise Exception("Failed optimization")


def set_shared_status(*args, progress, shared_data):
    # Patch _sim_monitor with this to access the shared_data variable in the
    # everserver main function.
    failed = len(
        [job for queue in progress for job in queue if job["status"] == JOB_FAILURE]
    )

    shared_data[SIM_PROGRESS_ENDPOINT] = {
        "status": {"failed": failed},
        "progress": progress,
    }


@pytest.mark.integration_test
def test_certificate_generation(copy_math_func_test_data_to_tmp):
    config = EverestConfig.load_file("config_minimal.yml")
    cert, key, pw = everserver._generate_certificate(
        ServerConfig.get_certificate_dir(config.output_dir)
    )

    # check that files are written
    assert os.path.exists(cert)
    assert os.path.exists(key)

    # check certificate is readable
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.load_cert_chain(cert, key, pw)  # raise on error


def test_hostfile_storage(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
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


@patch("sys.argv", ["name", "--config-file", "config_minimal.yml"])
@patch(
    "everest.detached.jobs.everserver._configure_loggers",
    side_effect=configure_everserver_logger,
)
def test_configure_logger_failure(mocked_logger, copy_math_func_test_data_to_tmp):
    config_file = "config_minimal.yml"
    config = EverestConfig.load_file(config_file)
    everserver.main()
    status = everserver_status(
        ServerConfig.get_everserver_status_path(config.output_dir)
    )

    assert status["status"] == ServerStatus.failed
    assert "Exception: Configuring logger failed" in status["message"]


@patch("sys.argv", ["name", "--config-file", "config_minimal.yml"])
@patch("everest.detached.jobs.everserver._configure_loggers")
@patch("requests.get")
def test_status_running_complete(
    mocked_get, mocked_logger, copy_math_func_test_data_to_tmp
):
    config_file = "config_minimal.yml"
    config = EverestConfig.load_file(config_file)

    def mocked_server(url, verify, auth, timeout, proxies):
        if "/experiment_status" in url:
            return JSONResponse(
                everserver.ExperimentStatus(
                    exit_code=EverestExitCode.COMPLETED
                ).model_dump_json()
            )
        if "/shared_data" in url:
            return JSONResponse(
                jsonable_encoder(
                    {
                        SIM_PROGRESS_ENDPOINT: {},
                        STOP_ENDPOINT: False,
                    }
                )
            )
        resp = requests.Response()
        resp.status_code = 200
        return resp

    mocked_get.side_effect = mocked_server

    everserver.main()

    status = everserver_status(
        ServerConfig.get_everserver_status_path(config.output_dir)
    )

    assert status["status"] == ServerStatus.completed
    assert status["message"] == "Optimization completed."


@patch("sys.argv", ["name", "--config-file", "config_minimal.yml"])
@patch("everest.detached.jobs.everserver._configure_loggers")
@patch("requests.get")
def test_status_failed_job(mocked_get, mocked_logger, copy_math_func_test_data_to_tmp):
    config_file = "config_minimal.yml"
    config = EverestConfig.load_file(config_file)

    def mocked_server(url, verify, auth, timeout, proxies):
        if "/experiment_status" in url:
            return JSONResponse(
                everserver.ExperimentStatus(
                    exit_code=EverestExitCode.TOO_FEW_REALIZATIONS
                ).model_dump_json()
            )
        if "/shared_data" in url:
            return JSONResponse(
                jsonable_encoder(
                    {
                        SIM_PROGRESS_ENDPOINT: {
                            "status": {"failed": 3},
                            "progress": [
                                [
                                    {
                                        "name": "job1",
                                        "status": JOB_FAILURE,
                                        "error": "job 1 error 1",
                                    },
                                    {
                                        "name": "job1",
                                        "status": JOB_FAILURE,
                                        "error": "job 1 error 2",
                                    },
                                ],
                                [
                                    {
                                        "name": "job2",
                                        "status": JOB_SUCCESS,
                                        "error": "",
                                    },
                                    {
                                        "name": "job2",
                                        "status": JOB_FAILURE,
                                        "error": "job 2 error 1",
                                    },
                                ],
                            ],
                        },
                        STOP_ENDPOINT: False,
                    }
                )
            )
        resp = requests.Response()
        resp.status_code = 200
        return resp

    mocked_get.side_effect = mocked_server

    everserver.main()

    status = everserver_status(
        ServerConfig.get_everserver_status_path(config.output_dir)
    )

    # The server should fail and store a user-friendly message.
    assert status["status"] == ServerStatus.failed
    assert OPT_FAILURE_REALIZATIONS in status["message"]
    assert "job1 Failed with: job 1 error 1" in status["message"]
    assert "job1 Failed with: job 1 error 2" in status["message"]
    assert "job2 Failed with: job 2 error 1" in status["message"]


@patch("sys.argv", ["name", "--config-file", "config_minimal.yml"])
@patch("everest.detached.jobs.everserver._configure_loggers")
@patch("requests.get")
def test_status_exception(mocked_get, mocked_logger, copy_math_func_test_data_to_tmp):
    config_file = "config_minimal.yml"
    config = EverestConfig.load_file(config_file)

    def mocked_server(url, verify, auth, timeout, proxies):
        if "/experiment_status" in url:
            return JSONResponse(
                everserver.ExperimentStatus(
                    exit_code=EverestExitCode.EXCEPTION, message="Some message"
                ).model_dump_json()
            )
        if "/shared_data" in url:
            return JSONResponse(
                jsonable_encoder(
                    {
                        SIM_PROGRESS_ENDPOINT: {
                            "status": {},
                            "progress": [],
                        },
                        STOP_ENDPOINT: False,
                    }
                )
            )
        resp = requests.Response()
        resp.status_code = 200
        return resp

    mocked_get.side_effect = mocked_server

    everserver.main()
    status = everserver_status(
        ServerConfig.get_everserver_status_path(config.output_dir)
    )

    assert status["status"] == ServerStatus.failed
    assert "Some message" in status["message"]


@pytest.mark.integration_test
@pytest.mark.xdist_group(name="starts_everest")
@pytest.mark.timeout(120)
@patch("sys.argv", ["name", "--config-file", "config_minimal.yml"])
async def test_status_max_batch_num(copy_math_func_test_data_to_tmp):
    config = EverestConfig.load_file("config_minimal.yml")
    config_dict = {
        **config.model_dump(exclude_none=True),
        "optimization": {"algorithm": "optpp_q_newton", "max_batch_num": 1},
    }
    config = EverestConfig.model_validate(config_dict)
    config.dump("config_minimal.yml")

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
@pytest.mark.timeout(120)
@patch("sys.argv", ["name", "--config-file", "config_minimal.yml"])
async def test_status_contains_max_runtime_failure(
    copy_math_func_test_data_to_tmp, min_config
):
    config_file = "config_minimal.yml"

    Path("SLEEP_job").write_text("EXECUTABLE sleep", encoding="utf-8")
    min_config["simulator"] = {"max_runtime": 2}
    min_config["forward_model"] = ["sleep 5"]
    min_config["install_jobs"] = [{"name": "sleep", "source": "SLEEP_job"}]

    tmp_config = EverestConfig(**min_config)
    tmp_config.dump(config_file)

    config = EverestConfig.load_file(config_file)

    await wait_for_server_to_complete(config)

    status = everserver_status(
        ServerConfig.get_everserver_status_path(config.output_dir)
    )

    assert status["status"] == ServerStatus.failed
    print(status["message"])
    assert (
        "sleep Failed with: The run is cancelled due to reaching MAX_RUNTIME"
        in status["message"]
    )
