import fileinput
from argparse import ArgumentParser

import pytest

from ert.__main__ import ert_parser
from ert.cli.main import ErtCliError, run_cli
from ert.mode_definitions import ENSEMBLE_SMOOTHER_MODE


@pytest.mark.usefixtures("copy_poly_case")
def test_running_smoother_raises_without_updateable_parameters():
    with fileinput.input("poly.ert", inplace=True) as fin:
        for line in fin:
            if "GEN_KW COEFFS coeff_priors" in line:
                print(f"{line[:-1]} UPDATE:FALSE")
            else:
                print(line, end="")

    parser = ArgumentParser(prog="test_main")
    parsed = ert_parser(
        parser,
        [
            ENSEMBLE_SMOOTHER_MODE,
            "--disable-monitor",
            "poly.ert",
        ],
    )

    with pytest.raises(ErtCliError) as e:
        run_cli(parsed)
    assert "All parameters are set to UPDATE:FALSE in" in str(e)
