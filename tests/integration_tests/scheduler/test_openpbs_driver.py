import os
from functools import partial
from pathlib import Path

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


@pytest.mark.timeout(30)
@pytest.mark.integration_test
@pytest.mark.usefixtures("copy_poly_case")
@pytest.mark.parametrize(
    "text_to_ignore",
    [
        "pbs_iff: cannot connect to host\npbs_iff: all reserved ports in use",
        "qstat: Invalid credential",
    ],
)
def test_that_openpbs_driver_ignores_qstat_flakiness(
    text_to_ignore, caplog, capsys, create_mock_flaky_qstat
):

    create_mock_flaky_qstat(text_to_ignore)
    with open("poly.ert", mode="a+", encoding="utf-8") as f:
        f.write("QUEUE_SYSTEM TORQUE\nNUM_REALIZATIONS 1")
    run_cli(
        ENSEMBLE_EXPERIMENT_MODE,
        "--enable-scheduler",
        "poly.ert",
    )
    assert Path("counter_file").exists()
    assert int(Path("counter_file").read_text(encoding="utf-8")) >= 3
    assert text_to_ignore not in capsys.readouterr().out
    assert text_to_ignore not in capsys.readouterr().err
    assert text_to_ignore not in caplog.text


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
