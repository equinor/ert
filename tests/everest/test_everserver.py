import json
import os
import ssl
from pathlib import Path
from unittest.mock import patch

from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, PlainTextResponse
from ropt.enums import OptimizerExitCode

from everest.config import EverestConfig, ServerConfig
from everest.detached import PROXY, ServerStatus, everserver_status
from everest.detached.jobs import everserver
from everest.detached.jobs.everest_server_api import (
    ExitCode,
    _generate_certificate,
    _write_hostfile,
)
from everest.simulator import JOB_FAILURE, JOB_SUCCESS
from everest.strings import (
    OPT_FAILURE_REALIZATIONS,
    SIM_PROGRESS_ENDPOINT,
    STOP_ENDPOINT,
)


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
        self._exit_code = OptimizerExitCode.TOO_FEW_REALIZATIONS
        return OptimizerExitCode.TOO_FEW_REALIZATIONS

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


def test_certificate_generation(copy_math_func_test_data_to_tmp):
    config = EverestConfig.load_file("config_minimal.yml")
    cert, key, pw = _generate_certificate(
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
    _write_hostfile(host_file_path, **expected_result)
    assert os.path.exists(host_file_path)
    with open(host_file_path, encoding="utf-8") as f:
        result = json.load(f)
    assert result == expected_result


@patch("sys.argv", ["name", "--config-file", "config_minimal.yml"])
@patch(
    "everest.detached.jobs.everserver._configure_loggers",
    side_effect=configure_everserver_logger,
)
def test_everserver_status_failure(mocked_logger, copy_math_func_test_data_to_tmp):
    config_file = "config_minimal.yml"
    config = EverestConfig.load_file(config_file)
    everserver.main()
    status = everserver_status(
        ServerConfig.get_everserver_status_path(config.output_dir)
    )

    assert status["status"] == ServerStatus.failed
    assert "Exception: Configuring logger failed" in status["message"]


import pytest
import requests


@pytest.mark.integration_test
@patch("sys.argv", ["name", "--config-file", "config_minimal.yml"])
@patch("everest.detached.jobs.everserver._configure_loggers")
@patch("everest.detached.jobs.everserver.export_to_csv")
@patch("requests.get")
def test_everserver_status_running_complete(
    mocked_get, mocked_export_to_csv, mocked_logger, copy_math_func_test_data_to_tmp
):
    config_file = "config_minimal.yml"
    config = EverestConfig.load_file(config_file)

    def mocked_server(url, verify, auth, proxies):
        if "/exit_code" in url:
            return JSONResponse(
                jsonable_encoder(
                    ExitCode(exit_code=OptimizerExitCode.OPTIMIZER_STEP_FINISHED)
                )
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

        return PlainTextResponse("Everest is running")

    mocked_get.side_effect = mocked_server

    everserver.main()

    status = everserver_status(
        ServerConfig.get_everserver_status_path(config.output_dir)
    )

    assert status["status"] == ServerStatus.completed


@patch("sys.argv", ["name", "--config-file", "config_minimal.yml"])
@patch("everest.detached.jobs.everserver._configure_loggers")
@patch("requests.get")
@patch("requests.post")
def test_everserver_status_failed_job(
    mocked_post,
    mocked_get,
    mocked_logger,
    copy_math_func_test_data_to_tmp,
):
    config_file = "config_minimal.yml"
    config = EverestConfig.load_file(config_file)

    def mocked_server(url, verify, auth, proxies):
        if "/exit_code" in url:
            return JSONResponse(
                jsonable_encoder(
                    ExitCode(exit_code=OptimizerExitCode.TOO_FEW_REALIZATIONS)
                )
            )
        if "/shared_data" in url:
            return JSONResponse(
                jsonable_encoder(
                    {
                        SIM_PROGRESS_ENDPOINT: {
                            "status": {"failed": 3},
                            "progress": [
                                [
                                    {"name": "job1", "status": JOB_FAILURE},
                                    {"name": "job1", "status": JOB_FAILURE},
                                ],
                                [
                                    {"name": "job2", "status": JOB_SUCCESS},
                                    {"name": "job2", "status": JOB_FAILURE},
                                ],
                            ],
                        },
                        STOP_ENDPOINT: False,
                    }
                )
            )
        return PlainTextResponse("Everest is running")

    mocked_get.side_effect = mocked_server

    mocked_post.side_effect = lambda url, verify, auth, proxies: PlainTextResponse("")

    everserver.main()

    url, cert, auth = ServerConfig.get_server_context(config.output_dir)
    requests.post(url + "/start", verify=cert, auth=auth, proxies=PROXY)  # type: ignore

    status = everserver_status(
        ServerConfig.get_everserver_status_path(config.output_dir)
    )

    # The server should fail and store a user-friendly message.
    assert status["status"] == ServerStatus.failed
    assert OPT_FAILURE_REALIZATIONS in status["message"]
    assert "3 job failures caused by: job1, job2" in status["message"]


@patch("sys.argv", ["name", "--config-file", "config_minimal.yml"])
@patch("everest.detached.jobs.everserver._configure_loggers")
@patch("requests.get")
@patch("requests.post")
def test_everserver_status_exception(
    mocked_post,
    mocked_get,
    mocked_logger,
    copy_math_func_test_data_to_tmp,
):
    config_file = "config_minimal.yml"
    config = EverestConfig.load_file(config_file)

    def mocked_server(url, verify, auth, proxies):
        if "/exit_code" in url:
            return JSONResponse(
                jsonable_encoder(ExitCode(message="Exception: Failed optimization"))
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
        return PlainTextResponse("Everest is running")

    mocked_get.side_effect = mocked_server

    mocked_post.side_effect = lambda url, verify, auth, proxies: PlainTextResponse("")

    everserver.main()

    url, cert, auth = ServerConfig.get_server_context(config.output_dir)
    requests.post(url + "/start", verify=cert, auth=auth, proxies=PROXY)  # type: ignore

    status = everserver_status(
        ServerConfig.get_everserver_status_path(config.output_dir)
    )

    # The server should fail, and store the exception that
    # start_optimization raised.
    assert status["status"] == ServerStatus.failed
    assert "Exception: Failed optimization" in status["message"]
