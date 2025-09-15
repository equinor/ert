from contextlib import ExitStack as does_not_raise
from datetime import datetime, timedelta

import hypothesis.strategies as st
import pytest
from hypothesis import assume, given
from pytest import MonkeyPatch, TempPathFactory
from resdata.summary import Summary

from ert.config import (
    ConfigValidationError,
    ConfigWarning,
    ErtConfig,
)
from ert.config.parsing import parse_observations

from .summary_generator import summaries


def make_observations(obs_config_contents, parse=True):
    obs_config_file = "obs_config"
    return ErtConfig.from_dict(
        {
            "NUM_REALIZATIONS": 1,
            "ECLBASE": "BASEBASEBASE",
            "SUMMARY": "*",
            "GEN_DATA": [["GEN", {"RESULT_FILE": "gen%d.txt", "REPORT_STEPS": "1"}]],
            "TIME_MAP": ("time_map.txt", "2020-01-01\n2020-01-02\n"),
            "OBS_CONFIG": (
                obs_config_file,
                parse_observations(obs_config_contents, obs_config_file)
                if parse
                else obs_config_contents,
            ),
        }
    ).observations


FOPR_VALUE = 1
FOPRH_VALUE = 2
SUMMARY_VALUES = {
    "FOPR": FOPR_VALUE,
    "FOPRH": FOPRH_VALUE,
}


def run_simulator(summary_values=SUMMARY_VALUES):
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
        for key, value in summary_values.items():
            t_step[key] = value

    summary.fwrite()


def make_refcase_observations(
    obs_config_contents, parse=True, extra_config=None, summary_values=SUMMARY_VALUES
):
    extra_config = extra_config or {}
    run_simulator(summary_values=summary_values)
    obs_config_file = "obs_config"
    return ErtConfig.from_dict(
        {
            "NUM_REALIZATIONS": 1,
            "ECLBASE": "BASEBASEBASE",
            "REFCASE": "MY_REFCASE",
            "SUMMARY": "*",
            "GEN_DATA": [["GEN", {"RESULT_FILE": "gen%d.txt", "REPORT_STEPS": "1"}]],
            "TIME_MAP": ("time_map.txt", "2020-01-01\n2020-01-02\n"),
            "OBS_CONFIG": (
                obs_config_file,
                parse_observations(obs_config_contents, obs_config_file)
                if parse
                else obs_config_contents,
            ),
            **extra_config,
        }
    ).observations


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.filterwarnings("ignore:Config contains a SUMMARY key")
def test_that_when_history_source_is_history_the_history_summary_vector_is_used():
    observations = make_refcase_observations(
        [("HISTORY_OBSERVATION", "FOPR")],
        extra_config={"HISTORY_SOURCE": "REFCASE_HISTORY"},
        parse=False,
    )
    assert list(observations["summary"]["observations"]) == [FOPRH_VALUE]


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.filterwarnings("ignore:Config contains a SUMMARY key")
def test_that_the_key_of_an_history_observation_must_be_in_the_refcase():
    with pytest.raises(
        ConfigValidationError, match="Key 'MISSINGH' is not present in refcase"
    ):
        make_refcase_observations(
            [("HISTORY_OBSERVATION", "MISSING")],
            parse=False,
        )


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.filterwarnings("ignore:Config contains a SUMMARY key")
def test_that_when_history_source_is_simulated_the_summary_vector_is_used():
    observations = make_refcase_observations(
        [("HISTORY_OBSERVATION", "FOPR")],
        extra_config={"HISTORY_SOURCE": "REFCASE_SIMULATED"},
        parse=False,
    )
    assert list(observations["summary"]["observations"]) == [FOPR_VALUE]


@pytest.mark.parametrize(
    "datestring, errors",
    [
        pytest.param("02.01.2020", True),
        pytest.param("02.1.2020", True),
        pytest.param("02-01-2020", True),
        pytest.param("02/01/2020", False),
    ],
)
@pytest.mark.filterwarnings("ignore:Config contains a SUMMARY key")
def test_date_parsing_in_observations(datestring, errors):
    observations = [
        (
            "SUMMARY_OBSERVATION",
            "FOPR",
            {"KEY": "FOPR", "VALUE": "1", "ERROR": "1", "DATE": datestring},
        )
    ]
    if errors:
        with pytest.raises(ValueError, match="Please use ISO date format"):
            make_observations(observations, parse=False)
    else:
        with pytest.warns(ConfigWarning, match="Please use ISO date format"):
            make_observations(observations, parse=False)


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


@given(
    summary=summaries(summary_keys=st.just(["FOPR", "FOPRH"])),
    value=st.floats(min_value=-1e9, max_value=1e9),
    data=st.data(),
)
def test_that_summary_observations_can_use_restart_for_index_if_refcase_is_given(
    tmp_path_factory: TempPathFactory, summary, value, data
):
    with MonkeyPatch.context() as patch:
        patch.chdir(tmp_path_factory.mktemp("history_observation_values_are_fetched"))
        smspec, unsmry = summary
        restart = data.draw(st.integers(min_value=1, max_value=len(unsmry.steps)))
        smspec.to_file("ECLIPSE_CASE.SMSPEC")
        unsmry.to_file("ECLIPSE_CASE.UNSMRY")
        observations = ErtConfig.from_dict(
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
                                "KEY": "FOPR",
                                "VALUE": str(value),
                                "ERROR": "1",
                                "RESTART": str(restart),
                            },
                        )
                    ],
                ),
            }
        ).observations["summary"]
        assert len(observations["time"]) == 1
        assert list(observations["observations"]) == pytest.approx([value])

        start_date = smspec.start_date.to_datetime()
        time_index = smspec.keywords.index("TIME    ")
        days = smspec.units[time_index] == "DAYS    "
        # start_date is considered to be restart=0, but that is not a step in
        # unsmry, therefore we need to look up restart-1
        restart_value = unsmry.steps[restart - 1].ministeps[-1].params[time_index]
        restart_time = start_date + (
            timedelta(days=float(restart_value))
            if days
            else timedelta(hours=float(restart_value))
        )

        assert abs(restart_time - observations["time"][0]) < timedelta(days=1.0)


def test_that_summary_observations_can_use_restart_for_index_if_time_map_is_given():
    restart = 1
    time_map = ["2024-01-01", "2024-02-02"]
    observations = ErtConfig.from_dict(
        {
            "ECLBASE": "ECLIPSE_CASE",
            "TIME_MAP": ("time_map.txt", "\n".join(time_map)),
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
                            "RESTART": str(restart),
                        },
                    )
                ],
            ),
        }
    ).observations["summary"]
    assert list(observations["time"]) == [datetime.fromisoformat(time_map[restart])]


def test_that_the_date_keyword_sets_the_summary_index_without_time_map_or_refcase():
    date = "2020-01-01"
    observations = ErtConfig.from_dict(
        {
            "ECLBASE": "ECLIPSE_CASE",
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
                            "DATE": date,
                        },
                    )
                ],
            ),
        }
    ).observations["summary"]
    assert list(observations["time"]) == [datetime.fromisoformat(date)]


@given(
    st.integers(min_value=0, max_value=10000), st.floats(min_value=-1e9, max_value=1e9)
)
def test_that_general_observations_can_use_restart_even_without_refcase_and_time_map(
    restart, value
):
    observations = ErtConfig.from_dict(
        {
            "GEN_DATA": [
                ["GEN", {"RESULT_FILE": "gen%d.txt", "REPORT_STEPS": str(restart)}]
            ],
            "OBS_CONFIG": (
                "obsconf",
                [
                    (
                        "GENERAL_OBSERVATION",
                        "OBS",
                        {
                            "DATA": "GEN",
                            "RESTART": str(restart),
                            "VALUE": str(value),
                            "ERROR": "1.0",
                        },
                    )
                ],
            ),
        }
    ).observations["gen_data"]
    assert list(observations["report_step"]) == [restart]
    assert list(observations["observations"]) == pytest.approx([value])


def test_that_the_date_keyword_sets_the_general_index_by_looking_up_time_map():
    restart = 1
    time_map = ["2024-01-01", "2024-02-02"]
    observations = ErtConfig.from_dict(
        {
            "TIME_MAP": ("time_map.txt", "\n".join(time_map)),
            "GEN_DATA": [
                ["GEN", {"RESULT_FILE": "gen%d.txt", "REPORT_STEPS": str(restart)}]
            ],
            "OBS_CONFIG": (
                "obsconf",
                [
                    (
                        "GENERAL_OBSERVATION",
                        "OBS",
                        {
                            "DATA": "GEN",
                            "DATE": time_map[restart],
                            "VALUE": "1.0",
                            "ERROR": "1.0",
                        },
                    )
                ],
            ),
        }
    ).observations["gen_data"]
    assert list(observations["report_step"]) == [restart]


@given(summary=summaries(), data=st.data())
def test_that_the_date_keyword_sets_the_report_step_by_looking_up_refcase(
    tmp_path_factory: TempPathFactory, summary, data
):
    with MonkeyPatch.context() as patch:
        patch.chdir(tmp_path_factory.mktemp("history_observation_values_are_fetched"))
        smspec, unsmry = summary
        smspec.to_file("ECLIPSE_CASE.SMSPEC")
        unsmry.to_file("ECLIPSE_CASE.UNSMRY")
        start_date = smspec.start_date.to_datetime()
        time_index = smspec.keywords.index("TIME    ")
        days = smspec.units[time_index] == "DAYS    "
        time_map = [s.ministeps[-1].params[time_index] for s in unsmry.steps]
        time_map = [
            start_date,
            *[
                start_date
                + (timedelta(days=float(t)) if days else timedelta(hours=float(t)))
                for t in time_map
            ],
        ]
        assume(len(time_map) > 2)
        restart = data.draw(st.integers(min_value=2, max_value=len(time_map) - 1))
        observations = ErtConfig.from_dict(
            {
                "REFCASE": "ECLIPSE_CASE",
                "GEN_DATA": [
                    ["GEN", {"RESULT_FILE": "gen%d.txt", "REPORT_STEPS": str(restart)}]
                ],
                "OBS_CONFIG": (
                    "obsconf",
                    [
                        (
                            "GENERAL_OBSERVATION",
                            "OBS",
                            {
                                "DATA": "GEN",
                                "DATE": time_map[restart].isoformat(),
                                "VALUE": "1.0",
                                "ERROR": "1.0",
                            },
                        )
                    ],
                ),
            }
        ).observations["gen_data"]

        assert list(observations["report_step"]) == [restart]


@pytest.mark.parametrize("std", [-1.0, 0, 0.0])
def test_that_error_must_be_greater_than_zero_in_summary_observations(std):
    with pytest.raises(
        ConfigValidationError, match=r"must be given a positive value|strictly > 0"
    ):
        make_observations(
            [
                (
                    "SUMMARY_OBSERVATION",
                    "FOPR",
                    {
                        "KEY": "FOPR",
                        "VALUE": "1",
                        "DATE": "2020-01-02",
                        "ERROR": str(std),
                    },
                )
            ],
            parse=False,
        )


def test_that_computed_error_must_be_greater_than_zero_in_summary_observations():
    with pytest.raises(
        ConfigValidationError, match=r"must be given a positive value|strictly > 0"
    ):
        make_observations(
            [
                (
                    "SUMMARY_OBSERVATION",
                    "FOPR",
                    {
                        "KEY": "FOPR",
                        "VALUE": "0",  # ERROR becomes zero when mode is REL
                        "DATE": "2020-01-02",
                        "ERROR": "1.0",
                        "ERROR_MODE": "REL",
                    },
                )
            ],
            parse=False,
        )


@pytest.mark.usefixtures("use_tmpdir")
def test_that_absolute_error_must_be_greater_than_zero_in_history_observations():
    with pytest.raises(
        ConfigValidationError, match=r"must be given a positive value|strictly > 0"
    ):
        make_refcase_observations(
            [
                (
                    "HISTORY_OBSERVATION",
                    "FOPR",
                    {
                        "ERROR": "0.0",
                        "ERROR_MIN": "0.0",
                    },
                )
            ],
            parse=False,
        )


@pytest.mark.usefixtures("use_tmpdir")
def test_that_computed_error_must_be_greater_than_zero_in_history_observations():
    with pytest.raises(
        ConfigValidationError, match=r"must be given a positive value|strictly > 0"
    ):
        make_refcase_observations(
            [
                (
                    "HISTORY_OBSERVATION",
                    "FOPR",
                    {
                        "ERROR": "1.0",
                        "ERROR_MODE": "REL",
                    },
                )
            ],
            summary_values={
                "FOPR": FOPR_VALUE,
                "FOPRH": 0,  # ERROR becomes zero when mode is REL
            },
            parse=False,
        )


@pytest.mark.parametrize("std", [-1.0, 0, 0.0])
def test_that_error_must_be_greater_than_zero_in_general_observations(std):
    with pytest.raises(
        ConfigValidationError, match=r"must be given a positive value|strictly > 0"
    ):
        make_observations(
            [
                (
                    "GENERAL_OBSERVATION",
                    "OBS",
                    {
                        "DATA": "GEN",
                        "DATE": "2020-01-02",
                        "INDEX_LIST": "1",
                        "VALUE": "1.0",
                        "ERROR": str(std),
                    },
                )
            ],
            parse=False,
        )


def test_that_all_errors_in_general_observations_must_be_greater_than_zero(tmpdir):
    with tmpdir.as_cwd():
        with open("obs_data.txt", "w", encoding="utf-8") as fh:
            # First error value will be 0
            fh.writelines(f"{float(i)} {float(i)}\n" for i in range(5))
        with pytest.raises(
            ConfigValidationError, match=r"must be given a positive value|strictly > 0"
        ):
            make_observations(
                [
                    (
                        "GENERAL_OBSERVATION",
                        "OBS",
                        {
                            "DATA": "GEN",
                            "DATE": "2020-01-02",
                            "OBS_FILE": "obs_data.txt",
                        },
                    )
                ],
                parse=False,
            )


def test_that_error_mode_is_not_allowed_in_general_observations():
    with pytest.raises(ConfigValidationError, match=r"Unknown ERROR_MODE"):
        make_observations(
            [
                (
                    "GENERAL_OBSERVATION",
                    "OBS",
                    {
                        "DATA": "GEN",
                        "DATE": "2020-01-02",
                        "INDEX_LIST": "1",
                        "VALUE": "1.0",
                        "ERROR": "0.1",
                        "ERROR_MODE": "REL",
                    },
                )
            ],
            parse=False,
        )


def test_that_error_min_is_not_allowed_in_general_observations():
    with pytest.raises(ConfigValidationError, match=r"Unknown ERROR_MIN"):
        make_observations(
            [
                (
                    "GENERAL_OBSERVATION",
                    "OBS",
                    {
                        "DATA": "GEN",
                        "DATE": "2020-01-02",
                        "INDEX_LIST": "1",
                        "VALUE": "1.0",
                        "ERROR": "0.1",
                        "ERROR_MIN": "0.05",
                    },
                )
            ],
            parse=False,
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
        observations = make_observations(
            [
                (
                    "GENERAL_OBSERVATION",
                    "OBS",
                    {
                        "DATA": "GEN",
                        "INDEX_LIST": "0,2,4,6,8",
                        "DATE": "2020-01-02",
                        "OBS_FILE": "obs_data.txt",
                    },
                )
            ],
            parse=False,
        )
        assert list(observations["gen_data"]["index"]) == [0, 2, 4, 6, 8]


def test_that_invalid_time_map_file_raises_config_validation_error():
    with pytest.raises(ConfigValidationError, match="Could not read timemap file"):
        _ = ErtConfig.from_dict({"TIME_MAP": ("time_map.txt", "invalid")})


def test_that_index_file_is_read(tmpdir):
    with tmpdir.as_cwd():
        with open("obs_idx.txt", "w", encoding="utf-8") as fh:
            fh.write("0\n2\n4\n6\n8")
        with open("obs_data.txt", "w", encoding="utf-8") as fh:
            fh.writelines(f"{float(i)} 0.1\n" for i in range(5))
        observations = make_observations(
            [
                (
                    "GENERAL_OBSERVATION",
                    "OBS",
                    {
                        "DATA": "GEN",
                        "DATE": "2020-01-02",
                        "INDEX_FILE": "obs_idx.txt",
                        "OBS_FILE": "obs_data.txt",
                    },
                )
            ],
            parse=False,
        )
        assert list(observations["gen_data"]["index"]) == [0, 2, 4, 6, 8]


def test_that_non_existent_obs_file_is_invalid():
    with pytest.raises(
        expected_exception=ConfigValidationError,
        match="did not resolve to a valid path:\n OBS_FILE",
    ):
        make_observations(
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
            parse=False,
        )


def test_that_non_existent_time_map_file_is_invalid():
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


@pytest.mark.usefixtures("use_tmpdir")
def test_that_general_observation_cannot_contain_both_value_and_obs_file():
    with open("obs_idx.txt", "w", encoding="utf-8") as fh:
        fh.write("0\n2\n4\n6\n8")
    with open("obs_data.txt", "w", encoding="utf-8") as fh:
        fh.writelines(f"{float(i)} 0.1\n" for i in range(5))
    with pytest.raises(
        ConfigValidationError, match=r"cannot contain both VALUE.*OBS_FILE"
    ):
        make_observations(
            [
                (
                    "GENERAL_OBSERVATION",
                    "OBS",
                    {
                        "DATA": "GEN",
                        "DATE": "2020-01-02",
                        "INDEX_FILE": "obs_idx.txt",
                        "OBS_FILE": "obs_data.txt",
                        "VALUE": "1.0",
                        "ERROR": "0.1",
                    },
                )
            ],
            parse=False,
        )


def test_that_general_observation_must_contain_either_value_or_obs_file():
    with pytest.raises(
        ConfigValidationError, match=r"must contain either VALUE.*OBS_FILE"
    ):
        make_observations(
            [
                (
                    "GENERAL_OBSERVATION",
                    "OBS",
                    {
                        "DATA": "GEN",
                        "DATE": "2020-01-02",
                    },
                )
            ],
            parse=False,
        )


def test_that_non_numbers_in_obs_file_shows_informative_error_message(tmpdir):
    with tmpdir.as_cwd():
        with open("obs_data.txt", "w", encoding="utf-8") as fh:
            fh.write("not_an_int 0.1\n")
        with pytest.raises(
            expected_exception=ConfigValidationError,
            match=r"Failed to read OBS_FILE obs_data.txt: could not convert"
            " string 'not_an_int' to float64 at row 0, column 1",
        ):
            make_observations(
                [
                    (
                        "GENERAL_OBSERVATION",
                        "OBS",
                        {
                            "DATA": "GEN",
                            "INDEX_LIST": "0,2,4,6,8",
                            "DATE": "2020-01-02",
                            "OBS_FILE": "obs_data.txt",
                        },
                    )
                ],
                parse=False,
            )


@pytest.mark.usefixtures("use_tmpdir")
def test_that_the_number_of_columns_in_obs_file_cannot_change():
    with open("obs_data.txt", "w", encoding="utf-8") as fh:
        fh.writelines(f"{float(i)} 0.1\n" for i in range(5))
        fh.writelines("0.1\n")
    with pytest.raises(
        ConfigValidationError, match="the number of columns changed from 2 to 1"
    ):
        make_observations(
            [
                (
                    "GENERAL_OBSERVATION",
                    "OBS",
                    {
                        "DATA": "GEN",
                        "INDEX_LIST": "0,2,4,6,8",
                        "DATE": "2020-01-02",
                        "OBS_FILE": "obs_data.txt",
                    },
                )
            ],
            parse=False,
        )


@pytest.mark.usefixtures("use_tmpdir")
def test_that_the_number_of_values_in_obs_file_must_be_even():
    with open("obs_data.txt", "w", encoding="utf-8") as fh:
        fh.writelines(f"{float(i)} 0.1 0.1\n" for i in range(5))
    with pytest.raises(ConfigValidationError, match="Expected even number of values"):
        make_observations(
            [
                (
                    "GENERAL_OBSERVATION",
                    "OBS",
                    {
                        "DATA": "GEN",
                        "INDEX_LIST": "0,2,4,6,8",
                        "DATE": "2020-01-02",
                        "OBS_FILE": "obs_data.txt",
                    },
                )
            ],
            parse=False,
        )


def test_that_giving_both_index_file_and_index_list_raises_an_exception(tmpdir):
    with tmpdir.as_cwd():
        with open("obs_idx.txt", "w", encoding="utf-8") as fh:
            fh.write("0\n2\n4\n6\n8")
        with pytest.raises(
            expected_exception=ConfigValidationError,
            match="both INDEX_FILE and INDEX_LIST",
        ):
            make_observations(
                [
                    (
                        "GENERAL_OBSERVATION",
                        "OBS",
                        {
                            "DATA": "GEN",
                            "INDEX_LIST": "0,2,4,6,8",
                            "INDEX_FILE": "obs_idx.txt",
                            "DATE": "2020-01-02",
                            "VALUE": "0.0",
                            "ERROR": "0.0",
                        },
                    )
                ],
                parse=False,
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


@pytest.mark.filterwarnings("ignore:Config contains a SUMMARY key")
@given(
    std=st.floats(min_value=0.1, max_value=1.0e3),
    with_ext=st.booleans(),
    summary=summaries(summary_keys=st.just(["FOPR", "FOPRH"])),
)
def test_that_history_observations_values_are_fetched_from_refcase(
    tmp_path_factory: TempPathFactory, summary, with_ext, std
):
    with MonkeyPatch.context() as patch:
        patch.chdir(tmp_path_factory.mktemp("history_observation_values_are_fetched"))
        smspec, unsmry = summary
        smspec.to_file("ECLIPSE_CASE.SMSPEC")
        unsmry.to_file("ECLIPSE_CASE.UNSMRY")

        observations = ErtConfig.from_dict(
            {
                "ECLBASE": "ECLIPSE_CASE",
                "REFCASE": f"ECLIPSE_CASE{'.DATA' if with_ext else ''}",
                "OBS_CONFIG": (
                    "obsconf",
                    [
                        (
                            "HISTORY_OBSERVATION",
                            "FOPR",
                            {
                                "ERROR": str(std),
                                "ERROR_MODE": "ABS",
                            },
                        )
                    ],
                ),
            }
        ).observations["summary"]

        steps = len(unsmry.steps)
        assert list(observations["response_key"]) == ["FOPR"] * steps
        assert list(observations["observations"]) == pytest.approx(
            [
                s.ministeps[-1].params[smspec.keywords.index("FOPRH")]
                for s in unsmry.steps
            ]
        )
        assert list(observations["std"]) == pytest.approx([std] * steps)


@pytest.mark.usefixtures("use_tmpdir")
def test_that_obs_file_must_have_the_same_number_of_lines_as_the_index_file():
    with open("obs_idx.txt", "w", encoding="utf-8") as fh:
        fh.write("0\n2\n4\n6")  # Should have 5 lines
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
                                "INDEX_FILE": "obs_idx.txt",  # shorter than obs_file
                                "OBS_FILE": "obs_data.txt",
                            },
                        )
                    ],
                ),
            }
        )


def test_that_obs_file_must_have_the_same_number_of_lines_as_the_length_of_index_list(
    tmpdir,
):
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


def test_that_general_observations_data_must_match_a_gen_datas_name():
    with pytest.raises(
        ConfigValidationError,
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
        ).observations["summary"]

        assert list(observations["response_key"]) == ["FGPR", "FOPR", "FWPR"]
        assert list(observations["observations"]) == pytest.approx([15, 20, 25])
        assert list(observations["std"]) == pytest.approx([1.5, 0.2, 10000])


def test_that_duplicate_observation_names_are_invalid():
    with pytest.raises(ConfigValidationError, match="Duplicate observation name FOPR"):
        make_observations(
            [
                (
                    "SUMMARY_OBSERVATION",
                    "FOPR",
                    {
                        "KEY": "FOPR",
                        "DATE": "2017-11-09",
                        "VALUE": "1.0",
                        "ERROR": "0.1",
                    },
                ),
                (
                    "GENERAL_OBSERVATION",
                    "FOPR",
                    {
                        "DATA": "RES",
                        "INDEX_LIST": "0",
                        "DATE": "2017-11-09",
                        "VALUE": "0.0",
                        "ERROR": "0.0",
                    },
                ),
            ],
            parse=False,
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
                                "ERROR": "1.0",
                                ("SEGMENT", "SEG"): {
                                    "START": "5",
                                    "STOP": "10",
                                    "ERROR": "0.05",
                                },
                            },
                        )
                    ],
                ),
            }
        ).observations["summary"]

        # default error_min is 0.1
        # default error method is RELMIN
        # default error is 0.1
        assert list(observations["std"]) == pytest.approx([1.0] * 5 + [0.1] * 5)


@pytest.mark.filterwarnings("ignore:Config contains a SUMMARY key")
def test_that_summary_default_error_min_is_applied():
    observations = make_observations(
        [
            (
                "SUMMARY_OBSERVATION",
                "FOPR",
                {
                    "VALUE": "1",
                    "ERROR": "0.01",
                    "KEY": "FOPR",
                    "DATE": "2020-01-02",
                    "ERROR_MODE": "RELMIN",
                },
            )
        ],
        parse=False,
    )
    # default error_min is 0.1
    assert list(observations["summary"]["std"]) == pytest.approx([0.1])


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
