from datetime import datetime
from pathlib import Path
from textwrap import dedent

import pytest
from ecl.summary import EclSum

from ert._c_wrappers.enkf import EnkfObs, ErtConfig


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
    Path("observations_config").write_text(
        "HISTORY_OBSERVATION FOPR;", encoding="utf-8"
    )
    Path("config.ert").write_text(config_text + extra_config, encoding="utf-8")
    run_simulator()
    ert_config = ErtConfig.from_file("config.ert")
    observations = EnkfObs(
        ert_config.model_config.history_source,
        ert_config.model_config.time_map,
        ert_config.ensemble_config.refcase,
        ert_config.ensemble_config,
    )
    observations.load(
        ert_config.model_config.obs_config_file,
        ert_config.analysis_config.get_std_cutoff(),
    )
    assert [obs.getValue() for obs in observations["FOPR"]] == [expected]


@pytest.mark.parametrize(
    "datestring, deprecated",
    [
        pytest.param("20.01.2000", True, id="dd.mm.yyyy gives warning"),
        pytest.param("20.1.2000", True, id="dd.m.yyyy gives warning"),
        pytest.param("20-01-2000", True, id="dd-mm-yyyy gives warning"),
        pytest.param("2000-01-20", False, id="YYYY-MM-DD does not give_warning"),
    ],
)
@pytest.mark.usefixtures("use_tmpdir")
def test_date_parsing_in_observations(datestring, deprecated, capfd):
    config_text = dedent(
        """
        NUM_REALIZATIONS 1
        JOBNAME my_case%d
        REFCASE MY_REFCASE
        OBS_CONFIG observations_config
        """
    )
    Path("observations_config").write_text(
        "SUMMARY_OBSERVATION FOPR_1 "
        f"{{ KEY=FOPR; VALUE=1; ERROR=1; DATE={datestring}; }};",
        encoding="utf-8",
    )
    Path("config.ert").write_text(config_text, encoding="utf-8")
    run_simulator()
    ert_config = ErtConfig.from_file("config.ert")
    observations = EnkfObs(
        ert_config.model_config.history_source,
        ert_config.model_config.time_map,
        ert_config.ensemble_config.refcase,
        ert_config.ensemble_config,
    )
    observations.load(
        ert_config.model_config.obs_config_file,
        ert_config.analysis_config.get_std_cutoff(),
    )
    captured = capfd.readouterr()
    if deprecated:
        assert "is deprecated" in captured.err
        assert "Please use ISO date format" in captured.err
    else:
        assert "deprecat" not in captured.err.lower()
        assert "deprecat" not in captured.out.lower()
