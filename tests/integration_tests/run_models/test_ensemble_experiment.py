import fileinput

import pytest

from ert.cli.main import ErtCliError
from ert.mode_definitions import ENSEMBLE_EXPERIMENT_MODE

from ..run_cli import run_cli


@pytest.mark.usefixtures("copy_snake_oil_case")
@pytest.mark.integration_test
def test_running_with_error_in_the_observation_configuration():
    with fileinput.input("snake_oil.ert", inplace=True) as fin:
        for line in fin:
            if line.startswith("REFCASE"):
                continue
            if line.startswith("TIME_MAP"):
                continue
            if line.startswith("HISTORY_SOURCE"):
                continue
            print(line, end="")

    with fileinput.input("observations/observations.txt", inplace=True) as fin:
        for line in fin:
            if line.startswith("HISTORY_OBSERVATION FOPR"):
                line = """SUMMARY_OBSERVATION WOPR_OP1_999
{
    VALUE   = 0.115;
    ERROR   = 0.11;
    DATE    = 2020-04-04;
    KEY     = WOPR:OP1;
};
"""
            if "2015-06-13" in line:
                line = "RESTART = 199;\n"

            print(line, end="")

    with pytest.raises(ErtCliError) as exc:
        run_cli(
            ENSEMBLE_EXPERIMENT_MODE,
            "snake_oil.ert",
        )
    assert "Missing observations ['WOPR_OP1_999']" in str(exc.value)
