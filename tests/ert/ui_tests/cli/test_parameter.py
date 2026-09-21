import fileinput
from argparse import ArgumentParser

import pytest

from ert.__main__ import ert_parser
from ert.cli.main import ErtCliError, run_cli
from ert.mode_definitions import ENIF_MODE, ENSEMBLE_SMOOTHER_MODE, ES_MDA_MODE


@pytest.mark.usefixtures("copy_poly_case")
@pytest.mark.parametrize("mode", [ENSEMBLE_SMOOTHER_MODE, ENIF_MODE, ES_MDA_MODE])
def test_that_cli_update_runs_reject_configs_without_updatable_parameters(mode):
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
            mode,
            "--disable-monitoring",
            "poly.ert",
        ],
    )

    with pytest.raises(
        ErtCliError,
        match="No parameters to update: all configured parameters have updates "
        r"disabled \(UPDATE:FALSE\)\.",
    ):
        run_cli(parsed)
