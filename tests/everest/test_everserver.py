import os
import ssl
from functools import partial
from unittest.mock import patch

from ropt.enums import OptimizerExitCode
from seba_sqlite.snapshot import SebaSnapshot

from everest.config import EverestConfig
from everest.detached import ServerStatus, everserver_status
from everest.detached.jobs import everserver
from everest.simulator import JOB_FAILURE, JOB_SUCCESS
from everest.strings import OPT_FAILURE_REALIZATIONS, SIM_PROGRESS_ENDPOINT


def configure_everserver_logger(*args, **kwargs):
    """Mock exception raised"""
    raise Exception("Configuring logger failed")


def check_status(*args, **kwargs):
    status = everserver_status(args[0])
    assert status["status"] == kwargs["status"]


def fail_optimization(
    config, simulation_callback, optimization_callback, from_ropt=False
):
    # Patch start_optimization to raise a failed optimization callback. Also
    # call the provided simulation callback, which has access to the shared_data
    # variable in the eversever main function. Patch that callback to modify
    # shared_data (see set_shared_status() below).
    simulation_callback(None, None)
    if from_ropt:
        return OptimizerExitCode.TOO_FEW_REALIZATIONS

    raise Exception("Failed optimization")


def set_shared_status(context_status, event, shared_data, progress):
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
    everest_config = EverestConfig.load_file("config_minimal.yml")
    cert, key, pw = everserver._generate_certificate(everest_config)

    # check that files are written
    assert os.path.exists(cert)
    assert os.path.exists(key)

    # check certificate is readable
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.load_cert_chain(cert, key, pw)  # raise on error


def test_hostfile_storage(copy_math_func_test_data_to_tmp):
    config = EverestConfig.load_file("config_minimal.yml")

    expected_result = {
        "host": "hostname.1.2.3",
        "port": "5000",
        "cert": "/a/b/c.cert",
        "auth": "1234",
    }
    everserver._write_hostfile(config, **expected_result)
    result = config.server_info
    assert result == expected_result


@patch("sys.argv", ["name", "--config-file", "config_minimal.yml"])
@patch(
    "everest.detached.jobs.everserver.configure_logger",
    side_effect=configure_everserver_logger,
)
def test_everserver_status_failure(_1, copy_math_func_test_data_to_tmp):
    config_file = "config_minimal.yml"
    config = EverestConfig.load_file(config_file)
    everserver.main()
    status = everserver_status(config)

    assert status["status"] == ServerStatus.failed
    assert "Exception: Configuring logger failed" in status["message"]


@patch("sys.argv", ["name", "--config-file", "config_minimal.yml"])
@patch("everest.detached.jobs.everserver.configure_logger")
@patch("everest.detached.jobs.everserver._generate_authentication")
@patch(
    "everest.detached.jobs.everserver._generate_certificate",
    return_value=(None, None, None),
)
@patch(
    "everest.detached.jobs.everserver._find_open_port",
    return_value=42,
)
@patch(
    "everest.detached.jobs.everserver._write_hostfile",
    side_effect=partial(check_status, status=ServerStatus.starting),
)
@patch("everest.detached.jobs.everserver._everserver_thread")
@patch(
    "everest.detached.jobs.everserver.start_optimization",
    side_effect=partial(check_status, status=ServerStatus.running),
)
@patch("everest.detached.jobs.everserver.validate_export", return_value=([], False))
@patch(
    "everest.detached.jobs.everserver.export_to_csv",
    side_effect=partial(check_status, status=ServerStatus.exporting_to_csv),
)
def test_everserver_status_running_complete(
    _1, _2, _3, _4, _5, _6, _7, _8, _9, copy_math_func_test_data_to_tmp
):
    config_file = "config_minimal.yml"
    config = EverestConfig.load_file(config_file)
    everserver.main()
    status = everserver_status(config)

    assert status["status"] == ServerStatus.completed
    assert status["message"] == "Optimization completed."


@patch("sys.argv", ["name", "--config-file", "config_minimal.yml"])
@patch("everest.detached.jobs.everserver.configure_logger")
@patch("everest.detached.jobs.everserver._generate_authentication")
@patch(
    "everest.detached.jobs.everserver._generate_certificate",
    return_value=(None, None, None),
)
@patch(
    "everest.detached.jobs.everserver._find_open_port",
    return_value=42,
)
@patch("everest.detached.jobs.everserver._write_hostfile")
@patch("everest.detached.jobs.everserver._everserver_thread")
@patch(
    "everest.detached.jobs.everserver.start_optimization",
    side_effect=partial(fail_optimization, from_ropt=True),
)
@patch(
    "everest.detached.jobs.everserver._sim_monitor",
    side_effect=partial(
        set_shared_status,
        progress=[
            [
                {"name": "job1", "status": JOB_FAILURE},
                {"name": "job1", "status": JOB_FAILURE},
            ],
            [
                {"name": "job2", "status": JOB_SUCCESS},
                {"name": "job2", "status": JOB_FAILURE},
            ],
        ],
    ),
)
def test_everserver_status_failed_job(
    _1, _2, _3, _4, _5, _6, _7, _8, copy_math_func_test_data_to_tmp
):
    config_file = "config_minimal.yml"
    config = EverestConfig.load_file(config_file)
    everserver.main()
    status = everserver_status(config)

    # The server should fail and store a user-friendly message.
    assert status["status"] == ServerStatus.failed
    assert OPT_FAILURE_REALIZATIONS in status["message"]
    assert "3 job failures caused by: job1, job2" in status["message"]


@patch("sys.argv", ["name", "--config-file", "config_minimal.yml"])
@patch("everest.detached.jobs.everserver.configure_logger")
@patch("everest.detached.jobs.everserver._generate_authentication")
@patch(
    "everest.detached.jobs.everserver._generate_certificate",
    return_value=(None, None, None),
)
@patch(
    "everest.detached.jobs.everserver._find_open_port",
    return_value=42,
)
@patch("everest.detached.jobs.everserver._write_hostfile")
@patch("everest.detached.jobs.everserver._everserver_thread")
@patch(
    "everest.detached.jobs.everserver.start_optimization",
    side_effect=fail_optimization,
)
@patch(
    "everest.detached.jobs.everserver._sim_monitor",
    side_effect=partial(set_shared_status, progress=[]),
)
def test_everserver_status_exception(
    _1, _2, _3, _4, _5, _6, _7, _8, copy_math_func_test_data_to_tmp
):
    config_file = "config_minimal.yml"
    config = EverestConfig.load_file(config_file)
    everserver.main()
    status = everserver_status(config)

    # The server should fail, and store the exception that
    # start_optimization raised.
    assert status["status"] == ServerStatus.failed
    assert "Exception: Failed optimization" in status["message"]


@patch("sys.argv", ["name", "--config-file", "config_one_batch.yml"])
@patch("everest.detached.jobs.everserver.configure_logger")
@patch("everest.detached.jobs.everserver._generate_authentication")
@patch(
    "everest.detached.jobs.everserver._generate_certificate",
    return_value=(None, None, None),
)
@patch(
    "everest.detached.jobs.everserver._find_open_port",
    return_value=42,
)
@patch("everest.detached.jobs.everserver._write_hostfile")
@patch("everest.detached.jobs.everserver._everserver_thread")
@patch(
    "everest.detached.jobs.everserver._sim_monitor",
    side_effect=partial(set_shared_status, progress=[]),
)
def test_everserver_status_max_batch_num(
    _1, _2, _3, _4, _5, _6, _7, copy_math_func_test_data_to_tmp
):
    config_file = "config_one_batch.yml"
    config = EverestConfig.load_file(config_file)
    everserver.main()
    status = everserver_status(config)

    # The server should complete without error.
    assert status["status"] == ServerStatus.completed

    # Check that there is only one batch.
    snapshot = SebaSnapshot(config.optimization_output_dir).get_snapshot(
        filter_out_gradient=False, batches=None
    )
    assert {data.batch for data in snapshot.simulation_data} == {0}
