import pytest

from ert.cli import ENSEMBLE_EXPERIMENT_MODE
from tests.integration_tests.run_cli import run_cli

from .conftest import mock_bin


@pytest.fixture(autouse=True)
def mock_openpbs(pytestconfig, monkeypatch, tmp_path):
    if pytestconfig.getoption("openpbs"):
        # User provided --openpbs, which means we should use the actual OpenPBS
        # cluster without mocking anything.
        return
    mock_bin(monkeypatch, tmp_path)


@pytest.mark.timeout(30)
@pytest.mark.integration_test
@pytest.mark.usefixtures("copy_poly_case")
def test_openpbs_driver_with_poly_example():
    with open("poly.ert", mode="a+", encoding="utf-8") as f:
        f.write("QUEUE_SYSTEM TORQUE\nNUM_REALIZATIONS 2")
    run_cli(
        ENSEMBLE_EXPERIMENT_MODE,
        "--enable-scheduler",
        "poly.ert",
    )
