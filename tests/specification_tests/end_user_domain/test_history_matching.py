"""
Use case tests for the domain of history matching. The user for this case
is a end-user that is trying to perform history matching.
"""

from argparse import ArgumentParser

import numpy as np
import pytest

from ert import LibresFacade
from ert.__main__ import ert_parser
from ert._c_wrappers.enkf import EnKFMain, ResConfig
from ert.cli import ENSEMBLE_SMOOTHER_MODE
from ert.cli.main import run_cli
from ert.shared.hook_implementations.workflows.disable_parameters import (
    DisableParametersUpdate,
)


@pytest.mark.integration_test
def test_that_posterior_has_lower_variance_than_prior(copy_case):
    # AS A history matcher

    # GIVEN that I am running the polynomial example
    copy_case("poly_example")

    # WHEN I run ert with ensemble smoother
    parser = ArgumentParser(prog="test_main")
    parsed = ert_parser(
        parser,
        [
            ENSEMBLE_SMOOTHER_MODE,
            "--current-case",
            "default",
            "--target-case",
            "target",
            "poly.ert",
            "--port-range",
            "1024-65535",
        ],
    )
    run_cli(parsed)

    # THEN I expect the variance to reduce with each step
    facade = LibresFacade.from_config_file("poly.ert")
    df_default = facade.load_all_gen_kw_data("default")
    df_target = facade.load_all_gen_kw_data("target")
    assert (
        0
        < np.linalg.det(df_target.cov().to_numpy())
        < np.linalg.det(df_default.cov().to_numpy())
    )


@pytest.mark.usefixtures("copy_poly_case")
def test_that_a_parameter_can_be_disabled():
    # AS A history matcher

    # GIVEN that I am running ertrwith the parameter DONT_UPDATE_KW
    with open("poly.ert", "a") as fh:
        fh.writelines("GEN_KW DONT_UPDATE_KW template.txt kw.txt prior.txt")
    with open("template.txt", "w") as fh:
        fh.writelines("MY_KEYWORD <MY_KEYWORD>")
    with open("prior.txt", "w") as fh:
        fh.writelines("MY_KEYWORD NORMAL 0 1")
    ert = EnKFMain(ResConfig("poly.ert"))

    # pylint: disable=no-member
    # (pylint is unable to read the members of update_step objects)

    # THEN by default the parameter is part of the update step
    parameters = [
        parameter.name
        for parameter in ert.update_configuration.update_steps[0].parameters
    ]
    assert "DONT_UPDATE_KW" in parameters

    # WHEN the DISABLE_PARAMETERS workflow job is run with DONT_UPDATE_KW
    DisableParametersUpdate(ert).run("DONT_UPDATE_KW")

    # THEN the DONT_UPDATE_KW keyword is not part of the update step
    parameters = [
        parameter.name
        for parameter in ert.update_configuration.update_steps[0].parameters
    ]
    assert "DONT_UPDATE_KW" not in parameters
