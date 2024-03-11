import os
from functools import partial

import pytest

from ert.cli import ENSEMBLE_EXPERIMENT_MODE
from ert.cli.main import ErtCliError
from ert.scheduler.openpbs_driver import OpenPBSDriver
from tests.integration_tests.run_cli import run_cli

from .conftest import mock_bin


@pytest.fixture(autouse=True)
def mock_openpbs(pytestconfig, monkeypatch, tmp_path):
    if pytestconfig.getoption("openpbs"):
        # User provided --openpbs, which means we should use the actual OpenPBS
        # cluster without mocking anything.
        return
    mock_bin(monkeypatch, tmp_path)


@pytest.fixture()
def queue_name_config():
    if queue_name := os.getenv("_ERT_TESTS_DEFAULT_QUEUE_NAME"):
        return f"\nQUEUE_OPTION TORQUE QUEUE {queue_name}"
    return ""


@pytest.mark.integration_test
@pytest.mark.usefixtures("copy_poly_case")
def test_openpbs_driver_with_poly_example(queue_name_config):
    with open("poly.ert", mode="a+", encoding="utf-8") as f:
        f.write("QUEUE_SYSTEM TORQUE\nNUM_REALIZATIONS 2")
        f.write(queue_name_config)
    run_cli(
        ENSEMBLE_EXPERIMENT_MODE,
        "--enable-scheduler",
        "poly.ert",
    )


async def mock_failure(message, *args, **kwargs):
    raise RuntimeError(message)


@pytest.mark.integration_test
@pytest.mark.usefixtures("copy_poly_case")
def test_openpbs_driver_with_poly_example_failing_submit_fails_ert_and_propagates_exception_to_user(
    monkeypatch, caplog, queue_name_config
):
    monkeypatch.setattr(
        OpenPBSDriver, "submit", partial(mock_failure, "Submit job failed")
    )
    with open("poly.ert", mode="a+", encoding="utf-8") as f:
        f.write("QUEUE_SYSTEM TORQUE\nNUM_REALIZATIONS 2")
        f.write(queue_name_config)
    with pytest.raises(ErtCliError):
        run_cli(
            ENSEMBLE_EXPERIMENT_MODE,
            "--enable-scheduler",
            "poly.ert",
        )
    assert "RuntimeError: Submit job failed" in caplog.text


@pytest.mark.integration_test
@pytest.mark.usefixtures("copy_poly_case")
def test_openpbs_driver_with_poly_example_failing_poll_fails_ert_and_propagates_exception_to_user(
    monkeypatch, caplog, queue_name_config
):
    monkeypatch.setattr(
        OpenPBSDriver, "poll", partial(mock_failure, "Status polling failed")
    )
    with open("poly.ert", mode="a+", encoding="utf-8") as f:
        f.write("QUEUE_SYSTEM TORQUE\nNUM_REALIZATIONS 2")
        f.write(queue_name_config)
    with pytest.raises(ErtCliError):
        run_cli(
            ENSEMBLE_EXPERIMENT_MODE,
            "--enable-scheduler",
            "poly.ert",
        )
    assert "RuntimeError: Status polling failed" in caplog.text
