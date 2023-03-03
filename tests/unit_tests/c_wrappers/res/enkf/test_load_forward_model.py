import fileinput
import logging
import os
from datetime import datetime
from pathlib import Path
from textwrap import dedent

import pytest
from ecl.summary import EclSum

from ert._c_wrappers.enkf import EnKFMain, ErtConfig
from ert.libres_facade import LibresFacade


def run_simulator(time_step_count, start_date) -> EclSum:
    ecl_sum = EclSum.writer("SNAKE_OIL_FIELD", start_date, 10, 10, 10)

    ecl_sum.addVariable("FOPR", unit="SM3/DAY")
    ecl_sum.addVariable("FOPRH", unit="SM3/DAY")

    ecl_sum.addVariable("WOPR", wgname="OP1", unit="SM3/DAY")
    ecl_sum.addVariable("WOPRH", wgname="OP1", unit="SM3/DAY")

    mini_step_count = 10
    for report_step in range(time_step_count):
        for mini_step in range(mini_step_count):
            t_step = ecl_sum.addTStep(
                report_step + 1, sim_days=report_step * mini_step_count + mini_step
            )
            t_step["FOPR"] = 1
            t_step["WOPR:OP1"] = 2
            t_step["FOPRH"] = 3
            t_step["WOPRH:OP1"] = 4

    return ecl_sum


@pytest.mark.usefixtures("copy_snake_oil_case_storage")
def test_load_inconsistent_time_map_summary(caplog):
    """
    Checking that we dont util_abort, we fail the forward model instead
    """
    cwd = os.getcwd()

    # Get rid of GEN_DATA as we are only interested in SUMMARY
    with fileinput.input("snake_oil.ert", inplace=True) as fin:
        for line in fin:
            if line.startswith("GEN_DATA"):
                continue
            print(line, end="")

    ert_config = ErtConfig.from_file("snake_oil.ert")
    ert = EnKFMain(ert_config)
    facade = LibresFacade(ert)
    realisation_number = 0
    assert (
        facade.get_current_fs().getStateMap()[realisation_number].name
        == "STATE_HAS_DATA"
    )  # Check prior state

    # Create a result that is incompatible with the refcase
    run_path = Path("storage") / "snake_oil" / "runpath" / "realization-0" / "iter-0"
    os.chdir(run_path)
    ecl_sum = run_simulator(1, datetime(2000, 1, 1))
    ecl_sum.fwrite()
    os.chdir(cwd)

    realizations = [False] * facade.get_ensemble_size()
    realizations[realisation_number] = True
    with caplog.at_level(logging.ERROR):
        loaded = facade.load_from_forward_model("default_0", realizations, 0)
    assert (
        "Realization: 0, load failure: 2 inconsistencies in time_map, first: "
        "Time mismatch for step: 0, response time: 2000-01-01, reference case: "
        "2010-01-01, last: Time mismatch for step: 1, response time: 2000-01-10, "
        f"reference case: 2010-01-10 from: {run_path.absolute()}"
        "/SNAKE_OIL_FIELD.UNSMRY"
    ) in caplog.messages
    assert loaded == 0
    assert (
        facade.get_current_fs().getStateMap()[realisation_number].name
        == "STATE_LOAD_FAILURE"
    )  # Check that status is as expected


@pytest.mark.usefixtures("copy_snake_oil_case_storage")
def test_load_forward_model():
    """
    Checking that we are able to load from forward model
    """
    # Get rid of GEN_DATA it causes a failure to load from forward model
    with fileinput.input("snake_oil.ert", inplace=True) as fin:
        for line in fin:
            if line.startswith("GEN_DATA"):
                continue
            print(line, end="")

    ert_config = ErtConfig.from_file("snake_oil.ert")
    ert = EnKFMain(ert_config)
    facade = LibresFacade(ert)
    realisation_number = 0

    realizations = [False] * facade.get_ensemble_size()
    realizations[realisation_number] = True
    loaded = facade.load_from_forward_model("default_0", realizations, 0)
    assert loaded == 1
    assert (
        facade.get_current_fs().getStateMap()[realisation_number].name
        == "STATE_HAS_DATA"
    )  # Check that status is as expected


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize(
    "summary_configuration, expected",
    [
        pytest.param(
            None,
            (1, None),
            id=(
                "Checking that we are able to successfully load "
                "from forward model with eclipse case even though "
                "there is no eclipse file in the run path. This is "
                "because no SUMMARY is added to the config"
            ),
        ),
        pytest.param(
            "SUMMARY *",
            (0, "Could not find SUMMARY file"),
            id=(
                "Check that loading fails if we have configured"
                "SUMMARY but no summary is available in the run path"
            ),
        ),
    ],
)
def test_load_forward_model_summary(summary_configuration, expected, caplog):
    config_text = dedent(
        """
        NUM_REALIZATIONS 1
        ECLBASE SNAKE_OIL_FIELD
        REFCASE SNAKE_OIL_FIELD
        """
    )
    if summary_configuration:
        config_text += summary_configuration
    Path("config.ert").write_text(config_text, encoding="utf-8")
    # Create refcase
    ecl_sum = run_simulator(1, datetime(2014, 9, 10))
    ecl_sum.fwrite()

    ert_config = ErtConfig.from_file("config.ert")
    ert = EnKFMain(ert_config)
    run_context = ert.create_ensemble_context("prior", [True], iteration=0)
    ert.createRunPath(run_context)
    facade = LibresFacade(ert)
    with caplog.at_level(logging.ERROR):
        loaded = facade.load_from_forward_model("prior", [True], 0)
    expected_loaded, expected_log_message = expected
    assert loaded == expected_loaded
    if expected_log_message:
        assert expected_log_message in "".join(caplog.messages)
