import os
from contextlib import ExitStack as does_not_raise
from datetime import datetime, timedelta
from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from resdata.summary import Summary

from ert.config import (
    ConfigValidationError,
    ConfigWarning,
    EnkfObs,
    EnkfObservationImplementationType,
    ErtConfig,
    SummaryObservation,
)
from ert.config.general_observation import GenObservation
from ert.config.observation_vector import ObsVector
from ert.config.parsing.observations_parser import ObservationConfigError

from .config_dict_generator import config_generators


def run_simulator():
    """
    Create an ecl summary file, we have one value for FOPR (1) and a different
    for FOPRH (2) so we can assert on the difference.
    """
    summary = Summary.writer("MY_REFCASE", datetime(2000, 1, 1), 10, 10, 10)

    summary.add_variable("FOPR", unit="SM3/DAY")
    summary.add_variable("FOPRH", unit="SM3/DAY")

    mini_step_count = 10

    for mini_step in range(mini_step_count):
        t_step = summary.addTStep(1, sim_days=mini_step_count + mini_step)
        t_step["FOPR"] = 1
        t_step["FOPRH"] = 2

    summary.fwrite()


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
    observations = ErtConfig.from_file("config.ert").enkf_obs
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
    if errors:
        with pytest.raises(ValueError, match="Please use ISO date format"):
            ErtConfig.from_file("config.ert")
    else:
        with pytest.warns(ConfigWarning, match="Please use ISO date format"):
            ErtConfig.from_file("config.ert")


def test_observations(minimum_case):
    observations = ErtConfig.from_file(
        minimum_case.ert_config.user_config_file
    ).enkf_obs
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
    for index in range(1, count):
        value = index * 10.5
        std = index / 10.0
        observation_vector.observations[index] = SummaryObservation(
            summary_key, observation_key, value, std
        )
        assert observation_vector.observations[index].value == value
        values.append((index, value, std))

    test_vector = observations[observation_key]

    for index, node in enumerate(test_vector):
        assert isinstance(node, SummaryObservation)
        assert node.value == (index + 1) * 10.5

    assert observation_vector == test_vector
    for index, value, std in values:
        assert index in test_vector.observations

        summary_observation_node = test_vector.observations[index]

        assert summary_observation_node.value == value
        assert summary_observation_node.std == std
        assert summary_observation_node.summary_key == summary_key


@pytest.mark.parametrize("std", [-1.0, 0, 0.0])
def test_summary_obs_invalid_observation_std(std):
    with pytest.raises(ValueError, match="must be strictly > 0"):
        SummaryObservation("summary_key", "observation_key", 1.0, std)


@pytest.mark.parametrize("std", [[-1.0], [0], [0.0], [1.0, 0]])
def test_gen_obs_invalid_observation_std(std):
    with pytest.raises(ValueError, match="must be strictly > 0"):
        GenObservation(
            np.array(range(len(std))),
            np.array(std),
            np.array(range(len(std))),
            np.array(range(len(std))),
        )


@settings(max_examples=10)
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::ert.config.ConfigWarning")
@given(config_generators(use_eclbase=st.just(True)))
def test_that_enkf_obs_keys_are_ordered(tmp_path_factory, config_generator):
    with config_generator(tmp_path_factory) as config_values:
        observations = ErtConfig.from_dict(
            config_values.to_config_dict("test.ert", os.getcwd())
        ).enkf_obs
        for o in config_values.observations:
            assert o.name in observations
        assert sorted(set(o.name for o in config_values.observations)) == list(
            observations.datasets.keys()
        )


def test_that_empty_observations_file_causes_exception(tmpdir):
    with tmpdir.as_cwd():
        config = dedent(
            """
        JOBNAME my_name%d
        NUM_REALIZATIONS 10
        OBS_CONFIG observations
        """
        )
        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(config)
        with open("observations", "w", encoding="utf-8") as fh:
            fh.writelines("")

        with pytest.raises(
            expected_exception=ObservationConfigError,
            match="Empty observations file",
        ):
            ErtConfig.from_file("config.ert")


def test_that_having_no_refcase_but_history_observations_causes_exception(tmpdir):
    with tmpdir.as_cwd():
        config = dedent(
            """
        ECLBASE my_name%d
        NUM_REALIZATIONS 10
        OBS_CONFIG observations
        """
        )
        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(config)
        with open("observations", "w", encoding="utf-8") as fo:
            fo.writelines("HISTORY_OBSERVATION FOPR;")
        with open("time_map.txt", "w", encoding="utf-8") as fo:
            fo.writelines("2023-02-01")

        with pytest.raises(
            expected_exception=ObservationConfigError,
            match="Missing REFCASE or TIME_MAP",
        ):
            ErtConfig.from_file("config.ert")


def test_that_index_list_is_read(tmpdir):
    with tmpdir.as_cwd():
        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(
                dedent(
                    """
                    JOBNAME my_name%d
                    NUM_REALIZATIONS 10
                    OBS_CONFIG observations
                    GEN_DATA RES RESULT_FILE:out_%d REPORT_STEPS:0 INPUT_FORMAT:ASCII
                    TIME_MAP time_map.txt
                    """
                )
            )
        with open("obs_data.txt", "w", encoding="utf-8") as fh:
            for i in range(5):
                fh.write(f"{float(i)} 0.1\n")
        with open("time_map.txt", "w", encoding="utf-8") as fo:
            fo.writelines("2017-11-09")
        with open("observations", "w", encoding="utf-8") as fo:
            fo.writelines(
                dedent(
                    """
                    GENERAL_OBSERVATION OBS {
                       DATA       = RES;
                       INDEX_LIST = 0,2,4,6,8;
                       DATE       = 2017-11-09;
                       OBS_FILE   = obs_data.txt;
                    };""",
                )
            )

        observations = ErtConfig.from_file("config.ert").enkf_obs
        assert observations["OBS"].observations[0].indices.tolist() == [0, 2, 4, 6, 8]


def test_that_index_file_is_read(tmpdir):
    with tmpdir.as_cwd():
        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(
                dedent(
                    """
                    JOBNAME my_name%d
                    NUM_REALIZATIONS 10
                    OBS_CONFIG observations
                    GEN_DATA RES RESULT_FILE:out
                    """
                )
            )
        with open("obs_idx.txt", "w", encoding="utf-8") as fh:
            fh.write("0\n2\n4\n6\n8")
        with open("obs_data.txt", "w", encoding="utf-8") as fh:
            for i in range(5):
                fh.write(f"{float(i)} 0.1\n")
        with open("observations", "w", encoding="utf-8") as fo:
            fo.writelines(
                dedent(
                    """
                    GENERAL_OBSERVATION OBS {
                       DATA       = RES;
                       INDEX_FILE = obs_idx.txt;
                       OBS_FILE   = obs_data.txt;
                    };""",
                )
            )

        observations = ErtConfig.from_file("config.ert").enkf_obs
        assert observations["OBS"].observations[0].indices.tolist() == [0, 2, 4, 6, 8]


def test_that_missing_obs_file_raises_exception(tmpdir):
    with tmpdir.as_cwd():
        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(
                dedent(
                    """
                    JOBNAME my_name%d
                    NUM_REALIZATIONS 10
                    OBS_CONFIG observations
                    """
                )
            )
        with open("observations", "w", encoding="utf-8") as fo:
            fo.writelines(
                """
                GENERAL_OBSERVATION OBS {
                   DATA       = RES;
                   INDEX_LIST = 0,2,4,6,8;
                   RESTART    = 0;
                   OBS_FILE   = obs_data.txt;
                };
                """
            )

        with pytest.raises(
            expected_exception=ObservationConfigError,
            match="did not resolve to a valid path:\n OBS_FILE",
        ):
            ErtConfig.from_file("config.ert")


def test_that_missing_time_map_raises_exception(tmpdir):
    with tmpdir.as_cwd():
        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(
                dedent(
                    """
                    JOBNAME my_name%d
                    NUM_REALIZATIONS 10
                    OBS_CONFIG observations
                    GEN_DATA RES RESULT_FILE:out_%d REPORT_STEPS:0 INPUT_FORMAT:ASCII
                    """
                )
            )
        with open("obs_data.txt", "w", encoding="utf-8") as fh:
            fh.write("1 0.1\n")
        with open("observations", "w", encoding="utf-8") as fo:
            fo.writelines(
                dedent(
                    """
                    GENERAL_OBSERVATION OBS {
                       DATA       = RES;
                       INDEX_LIST = 0,2,4,6,8;
                       DATE       = 2017-11-09;
                       OBS_FILE   = obs_data.txt;
                    };
                    """,
                )
            )

        with pytest.raises(
            expected_exception=ObservationConfigError,
            match="TIME_MAP",
        ):
            ErtConfig.from_file("config.ert")


def test_that_badly_formatted_obs_file_shows_informative_error_message(tmpdir):
    with tmpdir.as_cwd():
        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(
                dedent(
                    """
                    JOBNAME my_name%d
                    NUM_REALIZATIONS 10
                    OBS_CONFIG observations
                    GEN_DATA RES RESULT_FILE:out_%d REPORT_STEPS:0 INPUT_FORMAT:ASCII
                    TIME_MAP time_map.txt
                    """
                )
            )
        with open("obs_data.txt", "w", encoding="utf-8") as fh:
            fh.write("not_an_int 0.1\n")
        with open("time_map.txt", "w", encoding="utf-8") as fo:
            fo.writelines("2017-11-09")
        with open("observations", "w", encoding="utf-8") as fo:
            fo.writelines(
                dedent(
                    """
                    GENERAL_OBSERVATION OBS {
                       DATA       = RES;
                       INDEX_LIST = 0,2,4,6,8;
                       DATE    = 2017-11-09;
                       OBS_FILE   = obs_data.txt;
                    };
                    """,
                )
            )

        with pytest.raises(
            expected_exception=ObservationConfigError,
            match=r"Failed to read OBS_FILE .*/obs_data.txt: could not convert"
            " string 'not_an_int' to float64 at row 0, column 1",
        ):
            ErtConfig.from_file("config.ert")


def test_that_giving_both_index_file_and_index_list_raises_an_exception(tmpdir):
    with tmpdir.as_cwd():
        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(
                dedent(
                    """
                    JOBNAME my_name%d
                    NUM_REALIZATIONS 10
                    OBS_CONFIG observations
                    GEN_DATA RES RESULT_FILE:out_%d REPORT_STEPS:0 INPUT_FORMAT:ASCII
                    TIME_MAP time_map.txt
                    """
                )
            )
        with open("obs_idx.txt", "w", encoding="utf-8") as fh:
            fh.write("0\n2\n4\n6\n8")
        with open("obs_data.txt", "w", encoding="utf-8") as fh:
            for i in range(9):
                fh.write(f"{float(i)} 0.1\n")
        with open("time_map.txt", "w", encoding="utf-8") as fo:
            fo.writelines("2017-11-09")
        with open("observations", "w", encoding="utf-8") as fo:
            fo.writelines(
                dedent(
                    """
                    GENERAL_OBSERVATION OBS {
                       DATA       = RES;
                       INDEX_LIST = 0,2,4,6,8;
                       INDEX_FILE = obs_idx.txt;
                       DATE    = 2017-11-09;
                       OBS_FILE   = obs_data.txt;
                    };
                    """,
                )
            )

        with pytest.raises(
            expected_exception=ObservationConfigError,
            match="both INDEX_FILE and INDEX_LIST",
        ):
            ErtConfig.from_file("config.ert")


def run_sim(start_date, keys=None, values=None, days=None):
    """
    Create a summary file, the contents of which are not important
    """
    keys = keys if keys else [("FOPR", "SM3/DAY", None)]
    values = {} if values is None else values
    days = [1] if days is None else days
    summary = Summary.writer("ECLIPSE_CASE", start_date, 3, 3, 3)
    for key, unit, wname in keys:
        summary.add_variable(key, unit=unit, wgname=wname)
    for i in days:
        t_step = summary.add_t_step(i, sim_days=i)
        for key, _, wname in keys:
            if wname is None:
                t_step[key] = values.get(key, 1)
            else:
                t_step[key + ":" + wname] = values.get(key, 1)
    summary.fwrite()


@pytest.mark.parametrize(
    "time_map_statement, time_map_creator",
    [
        ("REFCASE ECLIPSE_CASE", lambda: run_sim(datetime(2014, 9, 10))),
        (
            "TIME_MAP time_map.txt",
            lambda: Path("time_map.txt").write_text(
                "2014-09-10\n2014-09-11\n", encoding="utf-8"
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "time_unit, time_delta, expectation",
    [
        pytest.param(
            "DAYS", 1.000347222, does_not_raise(), id="30 seconds offset from 1 day"
        ),
        pytest.param(
            "DAYS", 0.999664355, does_not_raise(), id="~30 seconds offset from 1 day"
        ),
        pytest.param("DAYS", 1.0, does_not_raise(), id="1 day"),
        pytest.param(
            "DAYS",
            "2.0",
            pytest.raises(
                ObservationConfigError,
                match=r"Could not find 2014-09-12 00:00:00 \(DAYS=2.0\)"
                " in the time map for observation FOPR_1",
            ),
            id="Outside tolerance days",
        ),
        pytest.param("HOURS", 24.0, does_not_raise(), id="1 day in hours"),
        pytest.param(
            "HOURS",
            48.0,
            pytest.raises(
                ObservationConfigError,
                match=r"Could not find 2014-09-12 00:00:00 \(HOURS=48.0\)"
                " in the time map for observation FOPR_1",
            ),
            id="Outside tolerance hours",
        ),
        pytest.param("DATE", "2014-09-11", does_not_raise(), id="1 day in date"),
        pytest.param(
            "DATE",
            "2014-09-12",
            pytest.raises(
                ObservationConfigError,
                match=r"Could not find 2014-09-12 00:00:00 \(DATE=2014-09-12\)"
                " in the time map for observation FOPR_1",
            ),
            id="Outside tolerance in date",
        ),
    ],
)
def test_that_loading_summary_obs_with_days_is_within_tolerance(
    tmpdir,
    time_delta,
    expectation,
    time_unit,
    time_map_statement,
    time_map_creator,
):
    with tmpdir.as_cwd():
        config = dedent(
            f"""
        NUM_REALIZATIONS 2

        ECLBASE ECLIPSE_CASE
        {time_map_statement}
        OBS_CONFIG observations
        """
        )
        observations = dedent(
            f"""
        SUMMARY_OBSERVATION FOPR_1
        {{
        VALUE   = 0.1;
        ERROR   = 0.05;
        {time_unit} = {time_delta};
        KEY     = FOPR;
        }};
        """
        )

        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(config)
        with open("observations", "w", encoding="utf-8") as fh:
            fh.writelines(observations)

        # We create a reference case
        time_map_creator()

        with expectation:
            ErtConfig.from_file("config.ert")


def test_that_having_observations_on_starting_date_errors(tmpdir):
    date = datetime(2014, 9, 10)
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
        DATE    = {date.isoformat()};
        KEY     = FOPR;
        }};
        """
        )

        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(config)
        with open("observations", "w", encoding="utf-8") as fh:
            fh.writelines(observations)

        # We create a reference case
        run_sim(date)

        with pytest.raises(
            ConfigValidationError,
            match="not possible to use summary observations from the start",
        ):
            ErtConfig.from_file("config.ert")


@pytest.mark.parametrize(
    "start, stop, message",
    [
        (
            100,
            10,
            "Segment FIRST_YEAR start after stop",
        ),
        (
            50,
            100,
            "does not contain any time steps",
        ),
        (
            -1,
            1,
            "Segment FIRST_YEAR out of bounds",
        ),
        (
            1,
            1000,
            "Segment FIRST_YEAR out of bounds",
        ),
        (
            1,
            1000,
            "Segment FIRST_YEAR out of bounds",
        ),
    ],
)
def test_that_out_of_bounds_segments_are_truncated(tmpdir, start, stop, message):
    with tmpdir.as_cwd():
        config = dedent(
            """
        NUM_REALIZATIONS 2

        ECLBASE ECLIPSE_CASE
        REFCASE ECLIPSE_CASE
        OBS_CONFIG observations
        """
        )
        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(config)
        with open("observations", "w", encoding="utf-8") as fo:
            fo.writelines(
                dedent(
                    f"""
                    HISTORY_OBSERVATION FOPR
                    {{
                       ERROR       = 0.20;
                       ERROR_MODE  = RELMIN;
                       ERROR_MIN   = 100;

                       SEGMENT FIRST_YEAR
                       {{
                          START = {start};
                          STOP  = {stop};
                          ERROR = 0.50;
                          ERROR_MODE = REL;
                       }};
                    }};
                    """
                )
            )
        run_sim(
            datetime(2014, 9, 10),
            [("FOPR", "SM3/DAY", None), ("FOPRH", "SM3/DAY", None)],
        )

        with pytest.warns(ConfigWarning, match=message):
            ErtConfig.from_file("config.ert")


@pytest.mark.parametrize(
    "keys",
    [
        [("FOPR", "SM3/DAY", None), ("FOPRH", "SM3/DAY", None)],
        [("WWIR", "SM3/DAY", "WNAME"), ("WWIRH", "SM3/DAY", "WNAME")],
    ],
)
def test_that_history_observations_are_loaded(tmpdir, keys):
    with tmpdir.as_cwd():
        config = dedent(
            """
        NUM_REALIZATIONS 2

        ECLBASE ECLIPSE_CASE
        REFCASE ECLIPSE_CASE
        OBS_CONFIG observations
        """
        )
        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(config)
        key, _, wname = keys[0]
        local_name = key if wname is None else (key + ":" + wname)
        with open("observations", "w", encoding="utf-8") as fo:
            fo.writelines(
                dedent(
                    f"""
                    HISTORY_OBSERVATION  {local_name}
                    {{
                       ERROR       = 0.20;
                       ERROR_MODE  = RELMIN;
                       ERROR_MIN   = 100;
                    }};
                    """
                )
            )
        with open("time_map.txt", "w", encoding="utf-8") as fo:
            fo.writelines("2023-02-01")
        run_sim(datetime(2014, 9, 10), keys)

        ert_config = ErtConfig.from_file("config.ert")

        observations = ert_config.enkf_obs
        assert [o.observation_key for o in observations] == [local_name]
        assert observations[local_name].observations[datetime(2014, 9, 11)].value == 1.0
        assert observations[local_name].observations[datetime(2014, 9, 11)].std == 100.0


def test_that_different_length_values_fail(monkeypatch, copy_snake_oil_case_storage):
    obs_file = Path.cwd() / "observations" / "observations.txt"
    observations = """
        \nGENERAL_OBSERVATION CUSTOM_DIFF_1
    {
       DATA       = SNAKE_OIL_WPR_DIFF;
       INDEX_LIST = 200;  -- shorter than number of points
       RESTART    = 199;
       OBS_FILE   = wpr_diff_obs.txt;
    };
        """
    with obs_file.open(mode="a") as fin:
        fin.write(observations)

    with pytest.raises(ObservationConfigError, match="must be of equal length"):
        ErtConfig.from_file("snake_oil.ert")


def test_that_missing_ensemble_key_warns(tmpdir):
    with tmpdir.as_cwd():
        config = dedent(
            """
        JOBNAME my_name%d
        NUM_REALIZATIONS 10
        OBS_CONFIG observations
        """
        )
        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(config)
        with open("observations", "w", encoding="utf-8") as fo:
            fo.writelines(
                dedent(
                    """
                    GENERAL_OBSERVATION OBS {
                       DATA       = RES;
                       INDEX_LIST = 0,2,4,6,8;
                       RESTART    = 0;
                       VALUE   = 1;
                       ERROR   = 1;
                    };""",
                )
            )

        with pytest.warns(
            ConfigWarning,
            match="Ensemble key RES does not exist",
        ):
            ErtConfig.from_file("config.ert")


def test_that_report_step_mismatch_warns(tmpdir):
    with tmpdir.as_cwd():
        config = dedent(
            """
        JOBNAME my_name%d
        NUM_REALIZATIONS 10
        OBS_CONFIG observations
        GEN_DATA RES INPUT_FORMAT:ASCII REPORT_STEPS:1 RESULT_FILE:file%d
        """
        )
        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(config)
        with open("observations", "w", encoding="utf-8") as fo:
            fo.writelines(
                dedent(
                    """
                    GENERAL_OBSERVATION OBS {
                       DATA       = RES;
                       INDEX_LIST = 0,2,4,6,8;
                       RESTART    = 0;
                       VALUE   = 1;
                       ERROR   = 1;
                    };""",
                )
            )

        with pytest.warns(
            ConfigWarning,
            match="is not configured to load from report step",
        ):
            ErtConfig.from_file("config.ert")


def test_that_history_observation_errors_are_calculated_correctly(tmpdir):
    with tmpdir.as_cwd():
        config = dedent(
            """
        NUM_REALIZATIONS 2

        ECLBASE ECLIPSE_CASE
        REFCASE ECLIPSE_CASE
        OBS_CONFIG observations
        """
        )
        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(config)
        with open("observations", "w", encoding="utf-8") as fo:
            fo.writelines(
                dedent(
                    """
                    HISTORY_OBSERVATION  FOPR
                    {
                       ERROR       = 0.20;
                       ERROR_MODE  = ABS;
                    };
                    HISTORY_OBSERVATION  FGPR
                    {
                       ERROR       = 0.1;
                       ERROR_MODE  = REL;
                    };
                    HISTORY_OBSERVATION  FWPR
                    {
                       ERROR       = 0.1;
                       ERROR_MODE  = RELMIN;
                       ERROR_MIN = 10000;
                    };
                    """
                )
            )
        with open("time_map.txt", "w", encoding="utf-8") as fo:
            fo.writelines("2023-02-01")
        run_sim(
            datetime(2014, 9, 10),
            [
                (k, "SM3/DAY", None)
                for k in ["FOPR", "FWPR", "FOPRH", "FWPRH", "FGPR", "FGPRH"]
            ],
            {"FOPRH": 20, "FGPRH": 15, "FWPRH": 25},
        )

        ert_config = ErtConfig.from_file("config.ert")

        observations = ert_config.enkf_obs

        assert observations["FGPR"].observation_key == "FGPR"
        assert observations["FGPR"].observations[datetime(2014, 9, 11)].value == 15.0
        assert observations["FGPR"].observations[datetime(2014, 9, 11)].std == 1.5

        assert observations["FOPR"].observation_key == "FOPR"
        assert observations["FOPR"].observations[datetime(2014, 9, 11)].value == 20.0
        assert observations["FOPR"].observations[datetime(2014, 9, 11)].std == 0.2

        assert observations["FWPR"].observation_key == "FWPR"
        assert observations["FWPR"].observations[datetime(2014, 9, 11)].value == 25.0
        assert observations["FWPR"].observations[datetime(2014, 9, 11)].std == 10000


@pytest.mark.filterwarnings("ignore::ert.config.ConfigWarning")
def test_that_std_cutoff_is_applied(tmpdir):
    with tmpdir.as_cwd():
        config = dedent(
            """
        NUM_REALIZATIONS 2

        ECLBASE ECLIPSE_CASE
        REFCASE ECLIPSE_CASE
        STD_CUTOFF 0.1
        OBS_CONFIG observations
        """
        )
        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(config)
        with open("observations", "w", encoding="utf-8") as fo:
            fo.writelines(
                dedent(
                    """
                    HISTORY_OBSERVATION  FOPR
                    {
                       ERROR       = 0.05;
                       ERROR_MODE  = ABS;
                    };
                    HISTORY_OBSERVATION  FGPR
                    {
                       ERROR       = 0.1;
                       ERROR_MODE  = REL;
                    };
                    """
                )
            )
        with open("time_map.txt", "w", encoding="utf-8") as fo:
            fo.writelines("2023-02-01")
        run_sim(
            datetime(2014, 9, 10),
            [(k, "SM3/DAY", None) for k in ["FOPR", "FOPRH", "FGPR", "FGPRH"]],
            {"FOPRH": 20, "FGPRH": 15},
        )

        ert_config = ErtConfig.from_file("config.ert")

        observations = ert_config.enkf_obs
        assert observations["FGPR"].observation_key == "FGPR"
        assert observations["FGPR"].observations[datetime(2014, 9, 11)].value == 15.0
        assert observations["FGPR"].observations[datetime(2014, 9, 11)].std == 1.5

        assert observations["FOPR"].observation_key == "FOPR"
        assert len(observations["FOPR"]) == 0


@pytest.mark.parametrize("obs_type", ["HISTORY_OBSERVATION", "SUMMARY_OBSERVATION"])
@pytest.mark.parametrize(
    "obs_content, match",
    [
        (
            "ERROR = -1;",
            'Failed to validate "-1"',
        ),
        (
            "ERROR_MODE=RELMIN; ERROR_MIN = -1; ERROR=1.0;",
            'Failed to validate "-1"',
        ),
        (
            "ERROR_MODE = NOT_ABS; ERROR=1.0;",
            'Failed to validate "NOT_ABS"',
        ),
    ],
)
def test_that_common_observation_error_validation_is_handled(
    tmpdir, obs_type, obs_content, match
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
        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(config)
        with open("observations", "w", encoding="utf-8") as fo:
            additional = (
                ""
                if obs_type == "HISTORY_OBSERVATION"
                else "RESTART = 1; VALUE=1.0; KEY = FOPR;"
            )
            fo.writelines(
                f"""
                    {obs_type}  FOPR
                    {{
                        {obs_content}
                        {additional}
                    }};
            """
            )
        with open("time_map.txt", "w", encoding="utf-8") as fo:
            fo.writelines("2023-02-01")
        run_sim(
            datetime(2014, 9, 10),
            [("FOPR", "SM3/DAY", None), ("FOPRH", "SM3/DAY", None)],
        )

        with pytest.raises(ObservationConfigError, match=match):
            ErtConfig.from_file("config.ert")


@pytest.mark.parametrize(
    "obs_content, match",
    [
        (
            dedent(
                """
                    SUMMARY_OBSERVATION  FOPR
                    {
                       DAYS       = -1;
                    };
                    """
            ),
            'Failed to validate "-1"',
        ),
        (
            dedent(
                """
                    SUMMARY_OBSERVATION  FOPR
                    {
                       VALUE       = exactly_1;
                    };
                    """
            ),
            'Failed to validate "exactly_1"',
        ),
        (
            dedent(
                """
                    SUMMARY_OBSERVATION  FOPR
                    {
                       DAYS       = 1;
                    };
                    """
            ),
            'Missing item "VALUE"',
        ),
        (
            dedent(
                """
                    SUMMARY_OBSERVATION  FOPR
                    {
                       KEY        = FOPR;
                       VALUE      = 2.0;
                       DAYS       = 1;
                    };
                    """
            ),
            'Missing item "ERROR"',
        ),
        (
            dedent(
                """
                    SUMMARY_OBSERVATION  FOPR
                    {
                       VALUE = 1;
                       ERROR = 0.1;
                    };
                    """
            ),
            'Missing item "KEY"',
        ),
        (
            dedent(
                """
                    HISTORY_OBSERVATION  FOPR
                    {
                       ERROR      = 0.1;

                       SEGMENT SEG
                       {
                          STOP  = 1;
                          ERROR = 0.50;
                       };
                    };
                    """
            ),
            'Missing item "START"',
        ),
        (
            dedent(
                """
                    HISTORY_OBSERVATION  FOPR
                    {
                       ERROR      = 0.1;

                       SEGMENT SEG
                       {
                          START  = 1;
                          ERROR = 0.50;
                       };
                    };
                    """
            ),
            'Missing item "STOP"',
        ),
        (
            dedent(
                """
                    HISTORY_OBSERVATION  FOPR
                    {
                       ERROR      = 0.1;

                       SEGMENT SEG
                       {
                          START = 0;
                          STOP  = 3.2;
                          ERROR = 0.50;
                       };
                    };
                    """
            ),
            'Failed to validate "3.2"',
        ),
        (
            dedent(
                """
                    HISTORY_OBSERVATION  FOPR
                    {
                       ERROR      = 0.1;

                       SEGMENT SEG
                       {
                          START = 1.1;
                          STOP  = 0;
                          ERROR = 0.50;
                       };
                    };
                    """
            ),
            'Failed to validate "1.1"',
        ),
        (
            dedent(
                """
                    HISTORY_OBSERVATION  FOPR
                    {
                       ERROR      = 0.1;

                       SEGMENT SEG
                       {
                          START = 1;
                          STOP  = 0;
                          ERROR = -1;
                       };
                    };
                    """
            ),
            'Failed to validate "-1"',
        ),
        (
            dedent(
                """
                    HISTORY_OBSERVATION  FOPR
                    {
                       ERROR      = 0.1;

                       SEGMENT SEG
                       {
                          START = 1;
                          STOP  = 0;
                          ERROR = 0.1;
                          ERROR_MIN = -1;
                       };
                    };
                    """
            ),
            'Failed to validate "-1"',
        ),
        (
            dedent(
                """
                    SUMMARY_OBSERVATION  FOPR
                    {
                       RESTART = -1;
                    };
                    """
            ),
            'Failed to validate "-1"',
        ),
        (
            dedent(
                """
                    SUMMARY_OBSERVATION  FOPR
                    {
                       RESTART = minus_one;
                    };
                    """
            ),
            'Failed to validate "minus_one"',
        ),
        (
            dedent(
                """
                    HISTORY_OBSERVATION  FOPR
                    {
                       ERROR      = 0.1;

                       SEGMENT SEG
                       {
                          START = 1;
                          STOP  = 0;
                          ERROR = 0.1;
                          ERROR_MODE = NOT_ABS;
                       };
                    };
                    """
            ),
            'Failed to validate "NOT_ABS"',
        ),
    ],
)
def test_that_summary_observation_validation_is_handled(tmpdir, obs_content, match):
    with tmpdir.as_cwd():
        config = dedent(
            """
        NUM_REALIZATIONS 2

        ECLBASE ECLIPSE_CASE
        REFCASE ECLIPSE_CASE
        OBS_CONFIG observations
        """
        )
        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(config)
        with open("observations", "w", encoding="utf-8") as fo:
            fo.writelines(obs_content)
        with open("time_map.txt", "w", encoding="utf-8") as fo:
            fo.writelines("2023-02-01")
        run_sim(
            datetime(2014, 9, 10),
            [("FOPR", "SM3/DAY", None), ("FOPRH", "SM3/DAY", None)],
        )

        with pytest.raises(ObservationConfigError, match=match):
            ErtConfig.from_file("config.ert")


@pytest.mark.parametrize(
    "obs_content, match",
    [
        (
            dedent(
                """
                    GENERAL_OBSERVATION  obs
                    {
                       DATA       = RES;
                       DATE       = 2023-02-01;
                       VALUE      = 1;
                    };
                    """
            ),
            "ERROR must also be given",
        ),
        (
            dedent(
                """
                    GENERAL_OBSERVATION  obs
                    {
                       DATE       = 2023-02-01;
                       VALUE      = 1;
                       ERROR      = 0.01;
                       ERROR_MIN  = 0.1;
                    };
                    """
            ),
            'Missing item "DATA"',
        ),
    ],
)
def test_validation_of_general_observation(tmpdir, obs_content, match):
    with tmpdir.as_cwd():
        config = dedent(
            """
        NUM_REALIZATIONS 2

        TIME_MAP time_map.txt
        OBS_CONFIG observations
        GEN_DATA RES RESULT_FILE:out_%d REPORT_STEPS:0 INPUT_FORMAT:ASCII
        """
        )
        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(config)
        with open("observations", "w", encoding="utf-8") as fo:
            fo.writelines(obs_content)
        with open("time_map.txt", "w", encoding="utf-8") as fo:
            fo.writelines("2023-02-01")

        with pytest.raises(ObservationConfigError, match=match):
            ErtConfig.from_file("config.ert")


@pytest.mark.parametrize(
    "observation_type",
    ["HISTORY_OBSERVATION", "SUMMARY_OBSERVATION", "GENERAL_OBSERVATION"],
)
def test_that_unknown_key_in_is_handled(tmpdir, observation_type):
    with tmpdir.as_cwd():
        config = dedent(
            """
        NUM_REALIZATIONS 2

        ECLBASE ECLIPSE_CASE
        REFCASE ECLIPSE_CASE
        OBS_CONFIG observations
        """
        )
        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(config)
        with open("observations", "w", encoding="utf-8") as fo:
            fo.writelines(f"{observation_type} FOPR {{SMERROR=0.1;DATA=key;}};")
        with open("time_map.txt", "w", encoding="utf-8") as fo:
            fo.writelines("2023-02-01")
        run_sim(
            datetime(2014, 9, 10),
            [("FOPR", "SM3/DAY", None), ("FOPRH", "SM3/DAY", None)],
        )

        with pytest.raises(ObservationConfigError, match="Unknown SMERROR"):
            ErtConfig.from_file("config.ert")


def test_validation_of_duplicate_names(
    tmpdir,
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
        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(config)
        with open("observations", "w", encoding="utf-8") as fo:
            fo.writelines(
                """SUMMARY_OBSERVATION FOPR {
                       KEY     = FOPR;
                       RESTART = 1;
                       VALUE   = 1.0;
                       ERROR   = 0.1;
                    };
                    HISTORY_OBSERVATION FOPR;
            """
            )
        with open("time_map.txt", "w", encoding="utf-8") as fo:
            fo.writelines("2023-02-01")
        run_sim(
            datetime(2014, 9, 10),
            [("FOPR", "SM3/DAY", None), ("FOPRH", "SM3/DAY", None)],
        )

        with pytest.raises(
            ObservationConfigError, match="Duplicate observation name FOPR"
        ):
            ErtConfig.from_file("config.ert")


def test_that_segment_defaults_are_applied(tmpdir):
    with tmpdir.as_cwd():
        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(
                dedent(
                    """
                    NUM_REALIZATIONS 2

                    ECLBASE ECLIPSE_CASE
                    REFCASE ECLIPSE_CASE
                    OBS_CONFIG observations
                    """
                )
            )
        with open("observations", "w", encoding="utf-8") as fo:
            fo.writelines(
                dedent(
                    """
                    HISTORY_OBSERVATION FOPR
                    {
                       SEGMENT SEG
                       {
                          START = 5;
                          STOP  = 9;
                          ERROR = 0.05;
                       };
                    };
                    """
                )
            )
        run_sim(
            datetime(2014, 9, 10),
            [("FOPR", "SM3/DAY", None), ("FOPRH", "SM3/DAY", None)],
            days=range(10),
        )

        observations = ErtConfig.from_file("config.ert").enkf_obs

        # default error_min is 0.1
        # default error method is RELMIN
        # default error is 0.1
        for i in range(1, 5):
            assert (
                observations["FOPR"]
                .observations[datetime(2014, 9, 11) + timedelta(days=i)]
                .std
                == 0.1
            )
        for i in range(5, 9):
            assert (
                observations["FOPR"]
                .observations[datetime(2014, 9, 11) + timedelta(days=i)]
                .std
                == 0.1
            )


def test_that_summary_default_error_min_is_applied(tmpdir):
    with tmpdir.as_cwd():
        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(
                dedent(
                    """
                    NUM_REALIZATIONS 2

                    ECLBASE ECLIPSE_CASE
                    REFCASE ECLIPSE_CASE
                    OBS_CONFIG observations
                    """
                )
            )
        with open("observations", "w", encoding="utf-8") as fo:
            fo.writelines(
                dedent(
                    """
                    SUMMARY_OBSERVATION FOPR
                    {
                        VALUE = 1;
                        ERROR = 0.01;
                        KEY = "FOPR";
                        RESTART = 1;
                        ERROR_MODE = RELMIN;
                    };
                    """
                )
            )
        run_sim(
            datetime(2014, 9, 10),
            [("FOPR", "SM3/DAY", None), ("FOPRH", "SM3/DAY", None)],
        )

        observations = ErtConfig.from_file("config.ert").enkf_obs

        # default error_min is 0.1
        assert observations["FOPR"].observations[datetime(2014, 9, 11)].std == 0.1


def test_unexpected_character_handling(tmpdir):
    with tmpdir.as_cwd():
        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(
                dedent(
                    """
                    NUM_REALIZATIONS 2
                    TIME_MAP time_map.txt
                    OBS_CONFIG observations
                    """
                )
            )
        with open("observations", "w", encoding="utf-8") as fo:
            fo.writelines(
                dedent(
                    """
                    GENERAL_OBSERVATION GEN_OBS
                    {
                       ERROR       $ 0.20;
                    };
                    """
                )
            )
        with open("time_map.txt", "w", encoding="utf-8") as fo:
            fo.writelines("2023-02-01")

        with pytest.raises(
            ValueError,
            match=r"Did not expect character: \$ \(on line 4:    ERROR       \$"
            r" 0.20;\). Expected one of {'EQUAL'}",
        ) as err_record:
            ErtConfig.from_file("config.ert")

        err = err_record.value.errors[0]
        assert err.line == 4
        assert err.end_line == 4
        assert err.column == 16
        assert err.end_column == 17
