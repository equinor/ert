from datetime import datetime
from textwrap import dedent

import pytest
from ecl.summary import EclSum

from ert._c_wrappers.enkf import EnkfObs, ResConfig


def run_simulator():
    """
    Create an ecl summary file, we have one value for FOPR (1) and a different
    for FOPRH (2) so we can assert on the difference.
    """
    ecl_sum = EclSum.writer("MY_REFCASE", datetime(2000, 1, 1), 10, 10, 10)

    ecl_sum.addVariable("FOPR", unit="SM3/DAY")
    ecl_sum.addVariable("FOPRH", unit="SM3/DAY")

    mini_step_count = 10

    for mini_step in range(mini_step_count):
        t_step = ecl_sum.addTStep(1, sim_days=mini_step_count + mini_step)
        t_step["FOPR"] = 1
        t_step["FOPRH"] = 2

    ecl_sum.fwrite()


@pytest.mark.parametrize(
    "extra_config, expected",
    [
        pytest.param("", 2.0, id="Default, equals REFCASE_HISTORY"),
        pytest.param(
            "HISTORY_SOURCE REFCASE_HISTORY",
            2.0,
            id="Expect to read the H post-fixed value, i.e. FOPRH",
        ),
        pytest.param(
            "HISTORY_SOURCE REFCASE_SIMULATED",
            1.0,
            id="Expect to read the actual value, i.e. FOPR",
        ),
    ],
)
@pytest.mark.usefixtures("use_tmpdir")
def test_that_correct_key_observation_is_loaded(extra_config, expected):
    config_text = dedent(
        """
        NUM_REALIZATIONS 1
        JOBNAME my_case%d
        REFCASE MY_REFCASE
        OBS_CONFIG observations_config
        """
    )
    with open("observations_config", "w") as fout:
        fout.write("HISTORY_OBSERVATION FOPR;")
    with open("config.ert", "w") as fout:
        fout.write(config_text + extra_config)
    run_simulator()
    res_config = ResConfig("config.ert")
    observations = EnkfObs(
        res_config.model_config.get_history_source(),
        res_config.model_config.get_time_map(),
        res_config.ecl_config.grid,
        res_config.ecl_config.refcase,
        res_config.ensemble_config,
    )
    observations.load(
        res_config.model_config.obs_config_file,
        res_config.analysis_config.getStdCutoff(),
    )
    assert [obs.getValue() for obs in observations["FOPR"]] == [expected]
