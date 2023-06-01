from contextlib import ExitStack as does_not_raise
from datetime import datetime
from pathlib import Path
from textwrap import dedent

import pytest
from ecl.summary import EclSum
from hypothesis import given
from hypothesis import strategies as st

from ert._c_wrappers.enkf import EnkfObs, ErtConfig, ObservationConfigError
from ert.parsing import ConfigValidationError, ConfigWarning

from .config_dict_generator import config_generators


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::ert.parsing.ConfigWarning")
@pytest.mark.usefixtures("set_site_config")
@given(config_generators(use_eclbase=st.just(True)))
def test_that_enkf_obs_keys_are_ordered(tmp_path_factory, config_generator):
    filename = "config.ert"
    with config_generator(tmp_path_factory, filename) as config_values:
        ert_config = ErtConfig.from_file(filename)
        observations = EnkfObs.from_ert_config(ert_config)
        for o in config_values.observations:
            assert observations.hasKey(o.name)
        assert sorted(set(o.name for o in config_values.observations)) == list(
            observations.getMatchingKeys("*")
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

        ert_config = ErtConfig.from_file("config.ert")

        with pytest.raises(
            expected_exception=ObservationConfigError,
            match="Empty observations file",
        ):
            _ = EnkfObs.from_ert_config(ert_config)


def test_that_having_no_refcase_but_history_observations_causes_exception(tmpdir):
    with tmpdir.as_cwd():
        config = dedent(
            """
        ECLBASE my_name%d
        NUM_REALIZATIONS 10
        OBS_CONFIG observations
        TIME_MAP time_map.txt
        """
        )
        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(config)
        with open("observations", "w", encoding="utf-8") as fo:
            fo.writelines("HISTORY_OBSERVATION FOPR;")
        with open("time_map.txt", "w", encoding="utf-8") as fo:
            fo.writelines("2023-02-01")

        ert_config = ErtConfig.from_file("config.ert")

        with pytest.raises(
            expected_exception=ObservationConfigError,
            match="REFCASE is required for HISTORY_OBSERVATION",
        ):
            _ = EnkfObs.from_ert_config(ert_config)


def test_that_missing_obs_file_raises_exception(tmpdir):
    with tmpdir.as_cwd():
        config = dedent(
            """
        JOBNAME my_name%d
        NUM_REALIZATIONS 10
        OBS_CONFIG observations
        TIME_MAP time_map.txt
        """
        )
        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(config)
        with open("observations", "w", encoding="utf-8") as fo:
            fo.writelines(
                [
                    "GENERAL_OBSERVATION OBS {",
                    "   DATA       = RES;",
                    "   INDEX_LIST = 0,2,4,6,8;",
                    "   RESTART    = 0;",
                    "   OBS_FILE   = obs_data.txt;",
                    "};",
                ]
            )
        with open("time_map.txt", "w", encoding="utf-8") as fo:
            fo.writelines("2023-02-01")

        ert_config = ErtConfig.from_file("config.ert")

        with pytest.raises(
            expected_exception=ObservationConfigError,
            match="did not resolve to a valid path:\n OBS_FILE",
        ):
            _ = EnkfObs.from_ert_config(ert_config)


def run_sim(start_date, keys=None, values=None):
    """
    Create a summary file, the contents of which are not important
    """
    keys = keys if keys else [("FOPR", "SM3/DAY", None)]
    values = {} if values is None else values
    ecl_sum = EclSum.writer("ECLIPSE_CASE", start_date, 3, 3, 3)
    for key, unit, wname in keys:
        ecl_sum.addVariable(key, unit=unit, wgname=wname)
    t_step = ecl_sum.addTStep(1, sim_days=1)
    for key, _, wname in keys:
        if wname is None:
            t_step[key] = values.get(key, 1)
        else:
            t_step[key + ":" + wname] = values.get(key, 1)
    ecl_sum.fwrite()


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
            _ = EnkfObs.from_ert_config(ErtConfig.from_file("config.ert"))


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
            _ = EnkfObs.from_ert_config(ErtConfig.from_file("config.ert"))


@pytest.mark.parametrize(
    "start, stop, message",
    [
        (
            100,
            10,
            "Segment start after stop",
        ),
        (
            50,
            100,
            "does not contain any time steps",
        ),
        (
            -1,
            1,
            "Segment out of bounds",
        ),
        (
            1,
            1000,
            "Segment out of bounds",
        ),
        (
            1,
            1000,
            "Segment out of bounds",
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
        with open("time_map.txt", "w", encoding="utf-8") as fo:
            fo.writelines("2023-02-01")
        run_sim(
            datetime(2014, 9, 10),
            [("FOPR", "SM3/DAY", None), ("FOPRH", "SM3/DAY", None)],
        )

        ert_config = ErtConfig.from_file("config.ert")

        with pytest.warns(ConfigWarning, match=message):
            _ = EnkfObs.from_ert_config(ert_config)


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

        observations = EnkfObs.from_ert_config(ert_config)
        assert [o.observation_key for o in observations] == [local_name]
        assert observations[local_name].observations[1].value == 1.0
        assert observations[local_name].observations[1].std == 100.0


def test_that_missing_time_map_raises_exception(tmpdir):
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
                       INDEX_LIST = 0,2,4,6,8;,
                       RESTART    = 1;
                       OBS_FILE   = obs_data.txt;
                    };""",
                )
            )
        with open("time_map.txt", "w", encoding="utf-8") as fo:
            fo.writelines("2023-02-01")

        ert_config = ErtConfig.from_file("config.ert")

        with pytest.raises(
            expected_exception=ObservationConfigError,
            match="TIMEMAP",
        ):
            _ = EnkfObs.from_ert_config(ert_config)


def test_that_missing_ensemble_key_warns(tmpdir):
    with tmpdir.as_cwd():
        config = dedent(
            """
        JOBNAME my_name%d
        NUM_REALIZATIONS 10
        OBS_CONFIG observations
        TIME_MAP time_map.txt
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
        with open("time_map.txt", "w", encoding="utf-8") as fo:
            fo.writelines("2023-02-01")

        ert_config = ErtConfig.from_file("config.ert")

        with pytest.warns(
            ConfigWarning,
            match="Ensemble key RES does not exist",
        ):
            _ = EnkfObs.from_ert_config(ert_config)


def test_that_report_step_mismatch_warns(tmpdir):
    with tmpdir.as_cwd():
        config = dedent(
            """
        JOBNAME my_name%d
        NUM_REALIZATIONS 10
        OBS_CONFIG observations
        TIME_MAP time_map.txt
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
        with open("time_map.txt", "w", encoding="utf-8") as fo:
            fo.writelines("2023-02-01")

        ert_config = ErtConfig.from_file("config.ert")

        with pytest.warns(
            ConfigWarning,
            match="is not configured to load from report step",
        ):
            _ = EnkfObs.from_ert_config(ert_config)


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

        observations = EnkfObs.from_ert_config(ert_config)

        assert observations["FGPR"].observation_key == "FGPR"
        assert observations["FGPR"].observations[1].value == 15.0
        assert observations["FGPR"].observations[1].std == 1.5

        assert observations["FOPR"].observation_key == "FOPR"
        assert observations["FOPR"].observations[1].value == 20.0
        assert observations["FOPR"].observations[1].std == 0.2

        assert observations["FWPR"].observation_key == "FWPR"
        assert observations["FWPR"].observations[1].value == 25.0
        assert observations["FWPR"].observations[1].std == 10000


@pytest.mark.filterwarnings("ignore::ert.parsing.ConfigWarning")
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

        observations = EnkfObs.from_ert_config(ert_config)
        assert observations["FGPR"].observation_key == "FGPR"
        assert observations["FGPR"].observations[1].value == 15.0
        assert observations["FGPR"].observations[1].std == 1.5

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

        ert_config = ErtConfig.from_file("config.ert")
        with pytest.raises(ObservationConfigError, match=match):
            _ = EnkfObs.from_ert_config(ert_config)


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
                          ERROR = 0.1
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

        ert_config = ErtConfig.from_file("config.ert")
        with pytest.raises(ObservationConfigError, match=match):
            _ = EnkfObs.from_ert_config(ert_config)
