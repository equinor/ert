from contextlib import ExitStack as does_not_raise
from datetime import datetime, timedelta

import pytest
from resdata.summary import Summary

from ert.config import (
    ConfigValidationError,
    ConfigWarning,
    ErtConfig,
    ObservationType,
    SummaryObservation,
)
from ert.config.general_observation import GenObservation
from ert.config.observation_vector import ObsVector
from ert.config.parsing import parse_observations


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
        pytest.param({}, 2.0, id="Default, equals REFCASE_HISTORY"),
        pytest.param(
            {"HISTORY_SOURCE": "REFCASE_HISTORY"},
            2.0,
            id="Expect to read the H post-fixed value, i.e. FOPRH",
        ),
        pytest.param(
            {"HISTORY_SOURCE": "REFCASE_SIMULATED"},
            1.0,
            id="Expect to read the actual value, i.e. FOPR",
        ),
    ],
)
@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.filterwarnings("ignore:Config contains a SUMMARY key")
def test_that_correct_key_observation_is_loaded(extra_config, expected):
    run_simulator()
    observations = ErtConfig.from_dict(
        {
            "ECLBASE": "my_case%d",
            "REFCASE": "MY_REFCASE",
            "OBS_CONFIG": ("obsconf", [("HISTORY_OBSERVATION", "FOPR")]),
            **extra_config,
        }
    ).enkf_obs
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
@pytest.mark.filterwarnings("ignore:Config contains a SUMMARY key")
def test_date_parsing_in_observations(datestring, errors):
    config_dict = {
        "ECLBASE": "my_case%d",
        "REFCASE": "MY_REFCASE",
        "OBS_CONFIG": (
            "obsconf",
            [
                (
                    "SUMMARY_OBSERVATION",
                    "FOPR_1 ",
                    {"KEY": "FOPR", "VALUE": "1", "ERROR": "1", "DATE": datestring},
                )
            ],
        ),
    }
    run_simulator()
    if errors:
        with pytest.raises(ValueError, match="Please use ISO date format"):
            ErtConfig.from_dict(config_dict)
    else:
        with pytest.warns(ConfigWarning, match="Please use ISO date format"):
            ErtConfig.from_dict(config_dict)


def test_that_using_summary_observations_without_eclbase_shows_user_error():
    with pytest.raises(ConfigValidationError, match="ECLBASE has to be set"):
        ErtConfig.from_dict(
            {
                "OBS_CONFIG": (
                    "obsconf",
                    [
                        (
                            "SUMMARY_OBSERVATION",
                            "FOPR_1",
                            {
                                "KEY": "FOPR",
                                "VALUE": "1",
                                "ERROR": "1",
                                "DATE": "2023-03-15",
                            },
                        )
                    ],
                )
            }
        )


def test_observations(minimum_case):
    observations = minimum_case.enkf_obs
    count = 10
    summary_key = "test_key"
    observation_key = "test_obs_key"
    observation_vector = ObsVector(
        ObservationType.SUMMARY,
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
            list(range(len(std))),
            list(std),
            list(range(len(std))),
            list(range(len(std))),
        )


def test_that_empty_observations_file_causes_exception():
    with pytest.raises(
        expected_exception=ConfigValidationError,
        match="Empty observations file",
    ):
        ErtConfig.from_dict({"OBS_CONFIG": ("obs_conf", "")})


@pytest.mark.filterwarnings("ignore:Config contains a SUMMARY key")
def test_that_having_no_refcase_but_history_observations_causes_exception():
    with pytest.raises(
        expected_exception=ConfigValidationError,
        match="REFCASE is required for HISTORY_OBSERVATION",
    ):
        ErtConfig.from_dict(
            {
                "ECLBASE": "my_name%d",
                "OBS_CONFIG": ("obsconf", [("HISTORY_OBSERVATION", "FOPR")]),
            }
        )


def test_that_index_list_is_read(tmpdir):
    with tmpdir.as_cwd():
        with open("obs_data.txt", "w", encoding="utf-8") as fh:
            fh.writelines(f"{float(i)} 0.1\n" for i in range(5))
        observations = ErtConfig.from_dict(
            {
                "GEN_DATA": [
                    [
                        "RES",
                        {"RESULT_FILE": "out"},
                    ]
                ],
                "OBS_CONFIG": (
                    "obsconf",
                    [
                        (
                            "GENERAL_OBSERVATION",
                            "OBS",
                            {
                                "DATA": "RES",
                                "INDEX_LIST": "0,2,4,6,8",
                                "OBS_FILE": "obs_data.txt",
                            },
                        )
                    ],
                ),
                "TIME_MAP": ("tm.txt", "2017-11-09"),
            }
        ).enkf_obs
        assert observations["OBS"].observations[0].indices == [0, 2, 4, 6, 8]


def test_that_invalid_time_map_file_raises_config_validation_error():
    with pytest.raises(ConfigValidationError, match="Could not read timemap file"):
        _ = ErtConfig.from_dict({"TIME_MAP": ("time_map.txt", "invalid")})


def test_that_index_file_is_read(tmpdir):
    with tmpdir.as_cwd():
        with open("obs_idx.txt", "w", encoding="utf-8") as fh:
            fh.write("0\n2\n4\n6\n8")
        with open("obs_data.txt", "w", encoding="utf-8") as fh:
            fh.writelines(f"{float(i)} 0.1\n" for i in range(5))
        observations = ErtConfig.from_dict(
            {
                "GEN_DATA": [
                    [
                        "RES",
                        {"RESULT_FILE": "out"},
                    ]
                ],
                "OBS_CONFIG": (
                    "obsconf",
                    [
                        (
                            "GENERAL_OBSERVATION",
                            "OBS",
                            {
                                "DATA": "RES",
                                "INDEX_FILE": "obs_idx.txt",
                                "OBS_FILE": "obs_data.txt",
                            },
                        )
                    ],
                ),
            }
        ).enkf_obs
        assert observations["OBS"].observations[0].indices == [0, 2, 4, 6, 8]


def test_that_missing_obs_file_raises_exception():
    with pytest.raises(
        expected_exception=ConfigValidationError,
        match="did not resolve to a valid path:\n OBS_FILE",
    ):
        ErtConfig.from_dict(
            {
                "OBS_CONFIG": (
                    "obsconf",
                    [
                        (
                            "GENERAL_OBSERVATION",
                            "OBS",
                            {
                                "DATA": "RES",
                                "INDEX_LIST": "0,2,4,6,8",
                                "RESTART": "0",
                                "OBS_FILE": "does_not_exist/at_all",
                            },
                        )
                    ],
                )
            }
        )


def test_that_missing_time_map_raises_exception():
    with pytest.raises(
        expected_exception=ConfigValidationError,
        match="TIME_MAP",
    ):
        ErtConfig.from_dict(
            {
                "GEN_DATA": [["RES", {"RESULT_FILE": "out"}]],
                "OBS_CONFIG": (
                    "obsconf",
                    [
                        (
                            "GENERAL_OBSERVATION",
                            "OBS",
                            {
                                "DATA": "RES",
                                "INDEX_LIST": "0",
                                "DATE": "2017-11-09",
                                "VALUE": "0.0",
                                "ERROR": "0.0",
                            },
                        )
                    ],
                ),
            }
        )


def test_that_badly_formatted_obs_file_shows_informative_error_message(tmpdir):
    with tmpdir.as_cwd():
        with open("obs_data.txt", "w", encoding="utf-8") as fh:
            fh.write("not_an_int 0.1\n")
        with pytest.raises(
            expected_exception=ConfigValidationError,
            match=r"Failed to read OBS_FILE obs_data.txt: could not convert"
            " string 'not_an_int' to float64 at row 0, column 1",
        ):
            ErtConfig.from_dict(
                {
                    "GEN_DATA": [["RES", {"RESULT_FILE": "out"}]],
                    "OBS_CONFIG": (
                        "obsconf",
                        [
                            (
                                "GENERAL_OBSERVATION",
                                "OBS",
                                {
                                    "DATA": "RES",
                                    "INDEX_LIST": "0,2,4,6,8",
                                    "OBS_FILE": "obs_data.txt",
                                },
                            )
                        ],
                    ),
                }
            )


def test_that_giving_both_index_file_and_index_list_raises_an_exception(tmpdir):
    with tmpdir.as_cwd():
        with open("obs_idx.txt", "w", encoding="utf-8") as fh:
            fh.write("0\n2\n4\n6\n8")
        with pytest.raises(
            expected_exception=ConfigValidationError,
            match="both INDEX_FILE and INDEX_LIST",
        ):
            ErtConfig.from_dict(
                {
                    "GEN_DATA": [["RES", {"RESULT_FILE": "out"}]],
                    "OBS_CONFIG": (
                        "obsconf",
                        [
                            (
                                "GENERAL_OBSERVATION",
                                "OBS",
                                {
                                    "DATA": "RES",
                                    "INDEX_LIST": "0,2,4,6,8",
                                    "INDEX_FILE": "obs_idx.txt",
                                    "VALUE": "0.0",
                                    "ERROR": "0.0",
                                },
                            )
                        ],
                    ),
                }
            )


def run_sim(start_date, keys=None, values=None, days=None):
    """
    Create a summary file, the contents of which are not important
    """
    keys = keys or [("FOPR", "SM3/DAY", None)]
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
        ({"REFCASE": "ECLIPSE_CASE"}, lambda: run_sim(datetime(2014, 9, 10))),
        (
            {"TIME_MAP": ("time_map.txt", "2014-09-10\n2014-09-11\n")},
            lambda: None,
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
                ConfigValidationError,
                match=r"Could not find 2014-09-12 00:00:00 \(DAYS=2.0\)"
                " in the time map for observations FOPR_1",
            ),
            id="Outside tolerance days",
        ),
        pytest.param("HOURS", 24.0, does_not_raise(), id="1 day in hours"),
        pytest.param(
            "HOURS",
            48.0,
            pytest.raises(
                ConfigValidationError,
                match=r"Could not find 2014-09-12 00:00:00 \(HOURS=48.0\)"
                " in the time map for observations FOPR_1",
            ),
            id="Outside tolerance hours",
        ),
        pytest.param("DATE", "2014-09-11", does_not_raise(), id="1 day in date"),
        pytest.param(
            "DATE",
            "2014-09-12",
            pytest.raises(
                ConfigValidationError,
                match=r"Could not find 2014-09-12 00:00:00 \(DATE=2014-09-12\)"
                " in the time map for observations FOPR_1",
            ),
            id="Outside tolerance in date",
        ),
    ],
)
@pytest.mark.filterwarnings("ignore:Config contains a SUMMARY key")
def test_that_loading_summary_obs_with_days_is_within_tolerance(
    tmpdir,
    time_delta,
    expectation,
    time_unit,
    time_map_statement,
    time_map_creator,
):
    with tmpdir.as_cwd():
        time_map_creator()

        with expectation:
            ErtConfig.from_dict(
                {
                    "ECLBASE": "ECLIPSE_CASE",
                    "OBS_CONFIG": (
                        "obsconf",
                        [
                            (
                                "SUMMARY_OBSERVATION",
                                "FOPR_1",
                                {
                                    "VALUE": "0.1",
                                    "ERROR": "0.05",
                                    time_unit: time_delta,
                                    "KEY": "FOPR",
                                },
                            )
                        ],
                    ),
                    **time_map_statement,
                }
            )


@pytest.mark.filterwarnings("ignore:Config contains a SUMMARY key")
def test_that_having_observations_on_starting_date_errors(tmpdir):
    date = datetime(2014, 9, 10)
    with tmpdir.as_cwd():
        # We create a reference case
        run_sim(date)

        with pytest.raises(
            ConfigValidationError,
            match="not possible to use summary observations from the start",
        ):
            ErtConfig.from_dict(
                {
                    "ECLBASE": "ECLIPSE_CASE",
                    "REFCASE": "ECLIPSE_CASE",
                    "OBS_CONFIG": (
                        "obsconf",
                        [
                            (
                                "SUMMARY_OBSERVATION",
                                "FOPR_1",
                                {
                                    "VALUE": "0.1",
                                    "ERROR": "0.05",
                                    "DATE": date.isoformat(),
                                    "KEY": "FOPR",
                                },
                            )
                        ],
                    ),
                }
            )


@pytest.mark.filterwarnings(
    r"ignore:.*Segment [^\s]+ "
    "((does not contain any time steps)|(out of bounds)|(start after stop)).*"
    ":ert.config.ConfigWarning"
)
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
@pytest.mark.filterwarnings("ignore:Config contains a SUMMARY key")
def test_that_out_of_bounds_segments_are_truncated(tmpdir, start, stop, message):
    with tmpdir.as_cwd():
        run_sim(
            datetime(2014, 9, 10),
            [("FOPR", "SM3/DAY", None), ("FOPRH", "SM3/DAY", None)],
        )

        with pytest.warns(ConfigWarning, match=message):
            ErtConfig.from_dict(
                {
                    "ECLBASE": "ECLIPSE_CASE",
                    "REFCASE": "ECLIPSE_CASE",
                    "OBS_CONFIG": (
                        "obsconf",
                        [
                            (
                                "HISTORY_OBSERVATION",
                                "FOPR",
                                {
                                    "ERROR": "0.20",
                                    "ERROR_MODE": "RELMIN",
                                    "ERROR_MIN": "100",
                                    ("SEGMENT", "FIRST_YEAR"): {
                                        "START": start,
                                        "STOP": stop,
                                        "ERROR": "0.50",
                                        "ERROR_MODE": "REL",
                                    },
                                },
                            )
                        ],
                    ),
                }
            )


@pytest.mark.parametrize("with_ext", [True, False])
@pytest.mark.parametrize(
    "keys",
    [
        [("FOPR", "SM3/DAY", None), ("FOPRH", "SM3/DAY", None)],
        [("WWIR", "SM3/DAY", "WNAME"), ("WWIRH", "SM3/DAY", "WNAME")],
    ],
)
@pytest.mark.filterwarnings("ignore:Config contains a SUMMARY key")
def test_that_history_observations_are_loaded(tmpdir, keys, with_ext):
    with tmpdir.as_cwd():
        key, _, wname = keys[0]
        local_name = key if wname is None else (key + ":" + wname)
        run_sim(datetime(2014, 9, 10), keys)

        ert_config = ErtConfig.from_dict(
            {
                "ECLBASE": "ECLIPSE_CASE",
                "REFCASE": f"ECLIPSE_CASE{'.DATA' if with_ext else ''}",
                "OBS_CONFIG": (
                    "obsconf",
                    [
                        (
                            "HISTORY_OBSERVATION",
                            local_name,
                            {
                                "ERROR": "0.20",
                                "ERROR_MODE": "RELMIN",
                                "ERROR_MIN": "100",
                            },
                        )
                    ],
                ),
            }
        )

        observations = ert_config.enkf_obs
        assert [o.observation_key for o in observations] == [local_name]
        assert observations[local_name].observations[datetime(2014, 9, 11)].value == 1.0
        assert observations[local_name].observations[datetime(2014, 9, 11)].std == 100.0


def test_that_different_length_values_fail(tmpdir):
    with tmpdir.as_cwd():
        with open("obs_data.txt", "w", encoding="utf-8") as fh:
            fh.writelines(f"{float(i)} 0.1\n" for i in range(5))
        with pytest.raises(ConfigValidationError, match="must be of equal length"):
            ErtConfig.from_dict(
                {
                    "GEN_DATA": [
                        [
                            "RES",
                            {"RESULT_FILE": "out"},
                        ]
                    ],
                    "OBS_CONFIG": (
                        "obsconf",
                        [
                            (
                                "GENERAL_OBSERVATION",
                                "OBS",
                                {
                                    "DATA": "RES",
                                    "INDEX_LIST": "200",  # shorter than obs_file
                                    "OBS_FILE": "obs_data.txt",
                                },
                            )
                        ],
                    ),
                }
            )


def test_that_missing_ensemble_key_warns():
    with pytest.warns(
        ConfigWarning,
        match="No GEN_DATA with name: RES found",
    ):
        ErtConfig.from_dict(
            {
                "OBS_CONFIG": (
                    "obsconf",
                    [
                        (
                            "GENERAL_OBSERVATION",
                            "OBS",
                            {
                                "DATA": "RES",
                                "INDEX_LIST": "0,2,4,6,8",
                                "RESTART": "0",
                                "VALUE": "1",
                                "ERROR": "1",
                            },
                        )
                    ],
                ),
            }
        )


def test_that_report_step_mismatch_warns():
    with pytest.warns(
        ConfigWarning,
        match="is not configured to load from report step",
    ):
        ErtConfig.from_dict(
            {
                "GEN_DATA": [
                    [
                        "RES",
                        {
                            "INPUT_FORMAT": "ASCII",
                            "REPORT_STEPS": "1",
                            "RESULT_FILE": "file%d",
                        },
                    ]
                ],
                "OBS_CONFIG": (
                    "obsconf",
                    [
                        (
                            "GENERAL_OBSERVATION",
                            "OBS",
                            {
                                "DATA": "RES",
                                "INDEX_LIST": "0,2,4,6,8",
                                "RESTART": "0",
                                "VALUE": "1",
                                "ERROR": "1",
                            },
                        )
                    ],
                ),
            }
        )


@pytest.mark.filterwarnings("ignore:Config contains a SUMMARY key")
def test_that_history_observation_errors_are_calculated_correctly(tmpdir):
    with tmpdir.as_cwd():
        run_sim(
            datetime(2014, 9, 10),
            [
                (k, "SM3/DAY", None)
                for k in ["FOPR", "FWPR", "FOPRH", "FWPRH", "FGPR", "FGPRH"]
            ],
            {"FOPRH": 20, "FGPRH": 15, "FWPRH": 25},
        )

        ert_config = ErtConfig.from_dict(
            {
                "ECLBASE": "ECLIPSE_CASE",
                "REFCASE": "ECLIPSE_CASE",
                "OBS_CONFIG": (
                    "obsconf",
                    [
                        (
                            "HISTORY_OBSERVATION",
                            "FOPR",
                            {
                                "ERROR": "0.20",
                                "ERROR_MODE": "ABS",
                            },
                        ),
                        (
                            "HISTORY_OBSERVATION",
                            "FGPR",
                            {
                                "ERROR": "0.1",
                                "ERROR_MODE": "REL",
                            },
                        ),
                        (
                            "HISTORY_OBSERVATION",
                            "FWPR",
                            {
                                "ERROR": "0.1",
                                "ERROR_MODE": "RELMIN",
                                "ERROR_MIN": "10000",
                            },
                        ),
                    ],
                ),
            }
        )

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


def test_validation_of_duplicate_names(tmpdir):
    with tmpdir.as_cwd():
        run_sim(
            datetime(2014, 9, 10),
            [("FOPR", "SM3/DAY", None), ("FOPRH", "SM3/DAY", None)],
        )

        with pytest.raises(
            ConfigValidationError, match="Duplicate observation name FOPR"
        ):
            ErtConfig.from_dict(
                {
                    "ECLBASE": "ECLIPSE_CASE",
                    "REFCASE": "ECLIPSE_CASE",
                    "OBS_CONFIG": (
                        "obsconf",
                        [
                            (
                                "SUMMARY_OBSERVATION",
                                "FOPR",
                                {
                                    "KEY": "FOPR",
                                    "RESTART": "1",
                                    "VALUE": "1.0",
                                    "ERROR": "0.1",
                                },
                            ),
                            ("HISTORY_OBSERVATION", "FOPR"),
                        ],
                    ),
                }
            )


@pytest.mark.filterwarnings("ignore:Config contains a SUMMARY key")
def test_that_segment_defaults_are_applied(tmpdir):
    with tmpdir.as_cwd():
        run_sim(
            datetime(2014, 9, 10),
            [("FOPR", "SM3/DAY", None), ("FOPRH", "SM3/DAY", None)],
            days=range(10),
        )

        observations = ErtConfig.from_dict(
            {
                "ECLBASE": "ECLIPSE_CASE",
                "REFCASE": "ECLIPSE_CASE",
                "OBS_CONFIG": (
                    "obsconf",
                    [
                        (
                            "HISTORY_OBSERVATION",
                            "FOPR",
                            {
                                ("SEGMENT", "SEG"): {
                                    "START": "5",
                                    "STOP": "9",
                                    "ERROR": "0.05",
                                },
                            },
                        )
                    ],
                ),
            }
        ).enkf_obs

        # default error_min is 0.1
        # default error method is RELMIN
        # default error is 0.1
        for i in range(1, 5):
            assert observations["FOPR"].observations[
                datetime(2014, 9, 11) + timedelta(days=i)
            ].std == pytest.approx(0.1)
        for i in range(5, 9):
            assert observations["FOPR"].observations[
                datetime(2014, 9, 11) + timedelta(days=i)
            ].std == pytest.approx(0.1)


@pytest.mark.filterwarnings("ignore:Config contains a SUMMARY key")
def test_that_summary_default_error_min_is_applied(tmpdir):
    with tmpdir.as_cwd():
        run_sim(
            datetime(2014, 9, 10),
            [("FOPR", "SM3/DAY", None), ("FOPRH", "SM3/DAY", None)],
        )

        observations = ErtConfig.from_dict(
            {
                "ECLBASE": "ECLIPSE_CASE",
                "REFCASE": "ECLIPSE_CASE",
                "OBS_CONFIG": (
                    "obsconf",
                    [
                        (
                            "SUMMARY_OBSERVATION",
                            "FOPR",
                            {
                                "VALUE": "1",
                                "ERROR": "0.01",
                                "KEY": "FOPR",
                                "RESTART": "1",
                                "ERROR_MODE": "RELMIN",
                            },
                        )
                    ],
                ),
            }
        ).enkf_obs

        # default error_min is 0.1
        assert observations["FOPR"].observations[datetime(2014, 9, 11)].std == 0.1


def make_observations(obs_config_contents):
    obs_config_file = "obs_config"
    return ErtConfig.from_dict(
        {
            "NUM_REALIZATIONS": 1,
            "ECLBASE": "BASEBASEBASE",
            "SUMMARY": "*",
            "GEN_DATA": [["GEN", {"RESULT_FILE": "gen.txt"}]],
            "TIME_MAP": ("time_map.txt", "2020-01-01\n"),
            "OBS_CONFIG": (
                obs_config_file,
                parse_observations(obs_config_contents, obs_config_file),
            ),
        }
    ).observations


def test_that_start_must_be_set_in_a_segment():
    with pytest.raises(ConfigValidationError, match='Missing item "START"'):
        make_observations("""
            HISTORY_OBSERVATION  FOPR {
               ERROR      = 0.1;

               SEGMENT SEG
               {
                  STOP  = 1;
                  ERROR = 0.50;
               };
            };
        """)


def test_that_stop_must_be_set_in_a_segment():
    with pytest.raises(ConfigValidationError, match='Missing item "STOP"'):
        make_observations("""
            HISTORY_OBSERVATION FOPR {
               ERROR      = 0.1;

               SEGMENT SEG {
                  START  = 1;
                  ERROR = 0.50;
               };
            };
        """)


def test_that_stop_must_be_given_integer_value():
    with pytest.raises(ConfigValidationError, match=r'Failed to validate "3\.2"'):
        make_observations("""
            HISTORY_OBSERVATION FOPR {
               ERROR      = 0.1;

               SEGMENT SEG
               {
                  START = 0;
                  STOP  = 3.2;
                  ERROR = 0.50;
               };
            };
        """)


def test_that_start_must_be_given_integer_value():
    with pytest.raises(ConfigValidationError, match=r'Failed to validate "1\.1"'):
        make_observations("""
            HISTORY_OBSERVATION FOPR {
               ERROR      = 0.1;

               SEGMENT SEG
               {
                  START = 1.1;
                  STOP  = 0;
                  ERROR = 0.50;
               };
            };
        """)


def test_that_error_must_be_positive_in_a_segment():
    with pytest.raises(ConfigValidationError, match='Failed to validate "-1"'):
        make_observations("""
            HISTORY_OBSERVATION  FOPR {
               ERROR      = 0.1;
               SEGMENT SEG {
                  START = 1;
                  STOP  = 0;
                  ERROR = -1;
               };
            };
        """)


def test_that_error_min_must_be_positive_in_a_segment():
    with pytest.raises(ConfigValidationError, match='Failed to validate "-1"'):
        make_observations("""
            HISTORY_OBSERVATION FOPR {
               ERROR      = 0.1;
               SEGMENT SEG {
                  START = 1;
                  STOP  = 0;
                  ERROR = 0.1;
                  ERROR_MIN = -1;
               };
            };
        """)


def test_that_error_mode_must_be_one_of_rel_abs_relmin_in_a_segment():
    with pytest.raises(ConfigValidationError, match='Failed to validate "NOT_ABS"'):
        make_observations("""
            HISTORY_OBSERVATION FOPR {
               ERROR      = 0.1;
               SEGMENT SEG
               {
                  START = 1;
                  STOP  = 0;
                  ERROR = 0.1;
                  ERROR_MODE = NOT_ABS;
               };
            };
        """)


def test_that_restart_must_be_positive_in_a_summary_observation():
    with pytest.raises(ConfigValidationError, match='Failed to validate "-1"'):
        make_observations("SUMMARY_OBSERVATION FOPR {RESTART = -1;};")


def test_that_restart_must_be_a_number_in_summary_observation():
    with pytest.raises(ConfigValidationError, match='Failed to validate "minus_one"'):
        make_observations("SUMMARY_OBSERVATION FOPR {RESTART = minus_one;};")


def test_that_value_must_be_set_in_summary_observation():
    with pytest.raises(ConfigValidationError, match='Missing item "VALUE"'):
        make_observations("SUMMARY_OBSERVATION FOPR {DAYS = 1;};")


def test_that_key_must_be_set_in_summary_observation():
    with pytest.raises(ConfigValidationError, match='Missing item "KEY"'):
        make_observations("""
            SUMMARY_OBSERVATION  FOPR {
               VALUE = 1;
               ERROR = 0.1;
            };
        """)


def test_that_data_must_be_set_in_general_observation():
    with pytest.raises(ConfigValidationError, match='Missing item "DATA"'):
        make_observations("""
            GENERAL_OBSERVATION obs {
               DATE       = 2023-02-01;
               VALUE      = 1;
               ERROR      = 0.01;
               ERROR_MIN  = 0.1;
            };
        """)


def test_that_error_must_be_a_positive_number_in_history_observation():
    with pytest.raises(ConfigValidationError, match='Failed to validate "-1"'):
        make_observations("HISTORY_OBSERVATION FOPR { ERROR = -1;};")


def test_that_error_min_must_be_a_positive_number_in_history_observation():
    with pytest.raises(ConfigValidationError, match='Failed to validate "-1"'):
        make_observations("""
            HISTORY_OBSERVATION FOPR {
                ERROR_MODE=RELMIN;
                ERROR_MIN = -1;
                ERROR=1.0;
            };
        """)


def test_that_error_mode_must_be_one_of_rel_abs_relmin_in_history_observation():
    with pytest.raises(ConfigValidationError, match='Failed to validate "NOT_ABS"'):
        make_observations("""
            HISTORY_OBSERVATION  FOPR {
                ERROR_MODE = NOT_ABS;
                ERROR=1.0;
            };
        """)


def test_that_error_must_be_a_positive_number_in_summary_observation():
    with pytest.raises(ConfigValidationError, match='Failed to validate "-1"'):
        make_observations("""
            SUMMARY_OBSERVATION  FOPR
            {
                ERROR = -1;
                RESTART = 1;
                VALUE=1.0;
                KEY = FOPR;
            };
        """)


def test_that_error_min_must_be_a_positive_number_in_summary_observation():
    with pytest.raises(ConfigValidationError, match='Failed to validate "-1"'):
        make_observations("""
            SUMMARY_OBSERVATION FOPR
            {
                ERROR_MODE=RELMIN;
                ERROR_MIN = -1;
                ERROR = 1.0;
                RESTART = 1;
                VALUE=1.0;
                KEY = FOPR;
            };
        """)


def test_that_error_mode_must_be_one_of_rel_abs_relmin_in_summary_observation():
    with pytest.raises(ConfigValidationError, match='Failed to validate "NOT_ABS"'):
        make_observations("""
            SUMMARY_OBSERVATION  FOPR
            {
                ERROR_MODE = NOT_ABS;
                ERROR=1.0;
                RESTART = 1;
                VALUE=1.0;
                KEY = FOPR;
            };
        """)


def test_that_days_must_be_a_positive_number_in_general_observation():
    with pytest.raises(ConfigValidationError, match='Failed to validate "-1"'):
        make_observations("""
            GENERAL_OBSERVATION FOPR
            {
                DAYS = -1;
                DATA = GEN;
            };
        """)


def test_that_hours_must_be_a_positive_number_in_general_observation():
    with pytest.raises(ConfigValidationError, match='Failed to validate "-1"'):
        make_observations("""
            GENERAL_OBSERVATION FOPR
            {
                HOURS = -1;
                DATA = GEN;
            };
        """)


def test_that_date_must_be_a_date_in_general_observation():
    with pytest.raises(ConfigValidationError, match="Please use ISO date format"):
        make_observations("""
            GENERAL_OBSERVATION FOPR
            {
                DATE = wednesday;
                VALUE = 1.0;
                ERROR = 0.1;
                DATA = GEN;
            };
        """)


def test_that_value_must_be_a_number_in_general_observation():
    with pytest.raises(ConfigValidationError, match='Failed to validate "exactly_1"'):
        make_observations("""
            GENERAL_OBSERVATION FOPR
            {
                VALUE = exactly_1;
                DATA = GEN;
            };
        """)


def test_that_error_must_be_set_in_general_observation():
    with pytest.raises(ConfigValidationError, match="ERROR"):
        make_observations("""
            GENERAL_OBSERVATION FOPR
            {
                VALUE = 1;
                DATA = GEN;
            };
        """)


def test_that_days_must_be_a_positive_number_in_summary_observation():
    with pytest.raises(ConfigValidationError, match='Failed to validate "-1"'):
        make_observations("""
            SUMMARY_OBSERVATION FOPR
            {
                DAYS = -1;
                KEY = FOPR;
            };
        """)


def test_that_hours_must_be_a_positive_number_in_summary_observation():
    with pytest.raises(ConfigValidationError, match='Failed to validate "-1"'):
        make_observations("""
            SUMMARY_OBSERVATION FOPR
            {
                HOURS = -1;
                KEY = FOPR;
            };
        """)


def test_that_date_must_be_a_date_in_summary_observation():
    with pytest.raises(ConfigValidationError, match="Please use ISO date format"):
        make_observations("""
            SUMMARY_OBSERVATION FOPR
            {
                DATE = wednesday;
                VALUE = 1.0;
                ERROR = 0.1;
                KEY = FOPR;
            };
        """)


def test_that_value_must_be_a_number_in_summary_observation():
    with pytest.raises(ConfigValidationError, match='Failed to validate "exactly_1"'):
        make_observations("""
            SUMMARY_OBSERVATION FOPR
            {
                VALUE = exactly_1;
                KEY = FOPR;
            };
        """)


def test_that_error_must_be_set_in_summary_observation():
    with pytest.raises(ConfigValidationError, match="ERROR"):
        make_observations("""
            SUMMARY_OBSERVATION FOPR
            {
                VALUE = 1;
                KEY = FOPR;
            };
        """)


@pytest.mark.parametrize(
    "observation_type",
    ["HISTORY_OBSERVATION", "SUMMARY_OBSERVATION", "GENERAL_OBSERVATION"],
)
def test_that_setting_an_unknown_key_is_not_valid(observation_type):
    with pytest.raises(ConfigValidationError, match="Unknown SMERROR"):
        make_observations(f"{observation_type} FOPR {{SMERROR=0.1;DATA=key;}};")


def test_that_setting_an_unknown_key_in_a_segment_is_not_valid():
    with pytest.raises(ConfigValidationError, match="Unknown SMERROR"):
        make_observations("""
            HISTORY_OBSERVATION FOPR {
                SEGMENT FIRST_YEAR {
                    START = 1;
                    STOP = 2;
                    SMERROR = 0.02;
                };
            };
        """)
