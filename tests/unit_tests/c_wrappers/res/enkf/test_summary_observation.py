import os
from contextlib import ExitStack as does_not_raise
from datetime import datetime
from textwrap import dedent

import pytest
from ecl.summary import EclSum

from ert._c_wrappers.enkf import (
    ActiveList,
    EnKFMain,
    ErtConfig,
    ObservationConfigError,
    SummaryObservation,
)


def test_create():
    sum_obs = SummaryObservation("WWCT:OP_X", "WWCT:OP_X", 0.25, 0.12)

    assert sum_obs.getValue() == 0.25
    assert sum_obs.getStandardDeviation() == 0.12
    assert sum_obs.getStdScaling() == 1.0


def test_std_scaling():
    sum_obs = SummaryObservation("WWCT:OP_X", "WWCT:OP_X", 0.25, 0.12)

    active_list = ActiveList()
    sum_obs.updateStdScaling(0.50, active_list)
    sum_obs.updateStdScaling(0.125, active_list)
    assert sum_obs.getStdScaling() == 0.125


def run_sim(start_date):
    """
    Create a summary file, the contents of which are not important
    """
    ecl_sum = EclSum.writer("ECLIPSE_CASE", start_date, 3, 3, 3)
    ecl_sum.addVariable("FOPR", unit="SM3/DAY")
    t_step = ecl_sum.addTStep(1, sim_days=1)
    t_step["FOPR"] = 1
    ecl_sum.fwrite()


@pytest.mark.parametrize(
    "time_delta, expectation",
    [
        pytest.param(
            "1.000347222", does_not_raise(), id="30 seconds offset from 1 day"
        ),
        pytest.param(
            "0.999664355", does_not_raise(), id="~30 seconds offset from 1 day"
        ),
        pytest.param("1.0", does_not_raise(), id="1 day"),
        pytest.param(
            "2.0",
            pytest.raises(
                ObservationConfigError,
                match="FOPR_1 does not have a matching time in the time map. DAYS=2",
            ),
            id="Outside tolerance",
        ),
    ],
)
def test_that_loading_summary_obs_with_days_is_within_tolerance(
    tmpdir,
    time_delta,
    expectation,
):
    with tmpdir.as_cwd():
        config = dedent(
            """
        NUM_REALIZATIONS 2

        ECLBASE ECLIPSE_CASE
        REFCASE ECLIPSE_CASE
        OBS_CONFIG observations
        """
        )
        observations = dedent(
            f"""
        SUMMARY_OBSERVATION FOPR_1
        {{
        VALUE   = 0.1;
        ERROR   = 0.05;
        DAYS    = {time_delta};
        KEY     = FOPR;
        }};
        """
        )

        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(config)
        with open("observations", "w", encoding="utf-8") as fh:
            fh.writelines(observations)

        # We create a reference case
        run_sim(datetime(2014, 9, 10))

        ert_config = ErtConfig.from_file("config.ert")
        os.chdir(ert_config.config_path)
        with expectation:
            ert = EnKFMain(ert_config)
            assert ert.getObservations().hasKey("FOPR_1")
