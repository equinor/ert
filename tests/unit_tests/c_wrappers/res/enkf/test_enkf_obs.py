from datetime import datetime
from pathlib import Path
from textwrap import dedent

import pytest
from ecl.summary import EclSum

from ert._c_wrappers.enkf import EnkfObs, ErtConfig, ObsVector
from ert._c_wrappers.enkf.enums import EnkfObservationImplementationType
from ert._c_wrappers.enkf.observations.summary_observation import SummaryObservation
from ert.parsing import ConfigWarning


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
        ECLBASE my_case%d
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
    observations = EnkfObs.from_ert_config(ert_config)
    assert [obs.value for obs in observations["FOPR"]] == [expected]


@pytest.mark.parametrize(
    "datestring, errors",
    [
        pytest.param("20.01.2000", True),
        pytest.param("20.1.2000", True),
        pytest.param("20-01-2000", True),
        pytest.param("20/01/2000", False),
    ],
)
@pytest.mark.usefixtures("use_tmpdir")
def test_date_parsing_in_observations(datestring, errors):
    config_text = dedent(
        """
        NUM_REALIZATIONS 1
        ECLBASE my_case%d
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
    if errors:
        with pytest.raises(ValueError, match="Please use ISO date format"):
            _ = EnkfObs.from_ert_config(ert_config)
    else:
        with pytest.warns(ConfigWarning, match="Please use ISO date format"):
            _ = EnkfObs.from_ert_config(ert_config)


def test_observations(minimum_case):
    observations = EnkfObs.from_ert_config(minimum_case.ert_config)
    count = 10
    summary_key = "test_key"
    observation_key = "test_obs_key"
    observation_vector = ObsVector(
        EnkfObservationImplementationType.SUMMARY_OBS,
        observation_key,
        "summary",
        {},
    )

    observations.obs_vectors[observation_key] = observation_vector

    values = []
    for index in range(0, count):
        value = index * 10.5
        std = index / 10.0
        observation_vector.observations[index] = SummaryObservation(
            summary_key, observation_key, value, std
        )
        assert observation_vector.observations[index].value == value
        values.append((index, value, std))

    test_vector = observations[observation_key]
    index = 0
    for node in test_vector:
        assert isinstance(node, SummaryObservation)
        assert node.value == index * 10.5
        index += 1

    assert observation_vector == test_vector
    for index, value, std in values:
        assert index in test_vector.observations

        summary_observation_node = test_vector.observations[index]

        assert summary_observation_node.value == value
        assert summary_observation_node.std == std
        assert summary_observation_node.summary_key == summary_key
