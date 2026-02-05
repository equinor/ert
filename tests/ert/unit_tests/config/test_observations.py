import logging
from contextlib import ExitStack as does_not_raise
from datetime import datetime, timedelta
from pathlib import Path
from textwrap import dedent
from typing import cast

import hypothesis.strategies as st
import polars as pl
import pytest
from hypothesis import assume, given
from polars.testing import assert_frame_equal
from resdata.summary import Summary
from resfo_utilities.testing import summaries

from ert.__main__ import run_convert_observations
from ert.config import (
    ConfigValidationError,
    ConfigWarning,
    ErtConfig,
)
from ert.config._create_observation_dataframes import create_observation_dataframes
from ert.config._observations import make_observations
from ert.config.parsing import parse_observations
from ert.config.parsing.observations_parser import (
    ObservationConfigError,
    ObservationType,
)
from ert.config.rft_config import RFTConfig
from ert.namespace import Namespace

pytestmark = pytest.mark.filterwarnings("ignore:Config contains a SUMMARY key")


def ert_config_from_parser(obs_config_contents):
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
                parse_observations(obs_config_contents, obs_config_file),
            ),
        }
    )


FOPR_VALUE = 1
FOPRH_VALUE = 2
SUMMARY_VALUES = {
    "FOPR": FOPR_VALUE,
    "FOPRH": FOPRH_VALUE,
}


def run_simulator(summary_values=SUMMARY_VALUES):
    """
    Create :term:`summary files` with one value for FOPR (1) and a different
    for FOPRH (2) so we can assert on the difference.
    """
    summary = Summary.writer("MY_REFCASE", datetime(2000, 1, 1), 10, 10, 10)

    for key in summary_values:
        summary.add_variable(key, unit="SM3/DAY")

    mini_step_count = 10

    for mini_step in range(mini_step_count):
        t_step = summary.addTStep(1, sim_days=mini_step_count + mini_step)
        for key, value in summary_values.items():
            t_step[key] = value

    summary.fwrite()


def make_refcase_observations(
    obs_config_contents, parse=True, extra_config=None, summary_values=SUMMARY_VALUES
):
    extra_config = extra_config or ""
    run_simulator(summary_values=summary_values)

    obs_config_file = "obs_config"
    Path(obs_config_file).write_text(obs_config_contents, encoding="utf-8")

    time_map_file = "time_map.txt"
    Path(time_map_file).write_text("2020-01-01\n2020-01-02\n", encoding="utf-8")

    config_content = (
        dedent(f"""
        NUM_REALIZATIONS 1
        ECLBASE BASEBASEBASE
        REFCASE MY_REFCASE
        SUMMARY *
        GEN_DATA GEN RESULT_FILE:gen%%d.txt REPORT_STEPS:1
        TIME_MAP {time_map_file}
        OBS_CONFIG {obs_config_file}
    """)
        + extra_config
    )
    Path("config.ert").write_text(config_content, encoding="utf-8")
    run_convert_observations(Namespace(config="config.ert"))

    migrated_config = ErtConfig.from_file("config.ert")
    return create_observation_dataframes(migrated_config.observation_declarations, None)


@pytest.mark.usefixtures("use_tmpdir")
def test_that_when_history_source_is_history_the_history_summary_vector_is_used():
    obs_config_contents = "HISTORY_OBSERVATION FOPR {};"
    observations = make_refcase_observations(
        obs_config_contents, extra_config="HISTORY_SOURCE REFCASE_HISTORY"
    )
    assert list(observations["summary"]["observations"]) == [FOPRH_VALUE]


@pytest.mark.usefixtures("use_tmpdir")
def test_that_the_key_of_an_history_observation_must_be_in_the_refcase():
    with pytest.raises(
        ConfigValidationError, match="Key 'MISSINGH' is not present in refcase"
    ):
        make_refcase_observations(
            "HISTORY_OBSERVATION MISSING {};",
        )


@pytest.mark.usefixtures("use_tmpdir")
def test_that_when_history_source_is_simulated_the_summary_vector_is_used():
    obs_config_contents = "HISTORY_OBSERVATION FOPR {};"
    observations = make_refcase_observations(
        obs_config_contents, extra_config="HISTORY_SOURCE REFCASE_SIMULATED"
    )
    assert list(observations["summary"]["observations"]) == [FOPR_VALUE]


@pytest.mark.parametrize(
    ("datestring", "errors"),
    [
        pytest.param("02.01.2020", True),
        pytest.param("02.1.2020", True),
        pytest.param("02-01-2020", True),
        pytest.param("02/01/2020", False),
    ],
)
def test_date_parsing_in_observations(datestring, errors):
    obs = [
        (
            {
                "type": ObservationType.SUMMARY,
                "name": "FOPR",
                "KEY": "FOPR",
                "VALUE": "1",
                "ERROR": "1",
                "DATE": datestring,
            }
        )
    ]
    if errors:
        with pytest.raises(ValueError, match="Please use ISO date format"):
            make_observations("", obs)
    else:
        with pytest.warns(ConfigWarning, match="Please use ISO date format"):
            make_observations("", obs)


def test_that_using_summary_observations_without_eclbase_shows_user_error():
    with pytest.raises(ConfigValidationError, match="ECLBASE has to be set"):
        ErtConfig.from_dict(
            {
                "OBS_CONFIG": (
                    "obsconf",
                    [
                        {
                            "type": ObservationType.SUMMARY,
                            "name": "FOPR_1",
                            "KEY": "FOPR",
                            "VALUE": "1",
                            "ERROR": "1",
                            "DATE": "2023-03-15",
                        }
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
    tmp_path_factory: pytest.TempPathFactory, summary, value, data
):
    with pytest.MonkeyPatch.context() as patch:
        tmp_dir = tmp_path_factory.mktemp("summary_obs_restart_migration")
        patch.chdir(tmp_dir)

        smspec, unsmry = summary
        restart = data.draw(st.integers(min_value=1, max_value=len(unsmry.steps)))
        smspec.to_file("ECLIPSE_CASE.SMSPEC")
        unsmry.to_file("ECLIPSE_CASE.UNSMRY")

        obs_config_content = dedent(
            f"""
            SUMMARY_OBSERVATION FOPR_1 {{
                KEY = FOPR;
                VALUE = {value};
                ERROR = 1;
                RESTART = {restart};
            }};
            """
        )
        (tmp_dir / "obs.conf").write_text(obs_config_content)

        config_content = dedent(
            """
            NUM_REALIZATIONS 1
            ECLBASE ECLIPSE_CASE
            REFCASE ECLIPSE_CASE
            OBS_CONFIG obs.conf
            """
        )
        config_file = tmp_dir / "config.ert"
        config_file.write_text(config_content)

        run_convert_observations(Namespace(config=str(config_file)))

        migrated_config = ErtConfig.from_file("config.ert")
        observations = create_observation_dataframes(
            migrated_config.observation_declarations, None
        )["summary"]

        assert len(observations["time"]) == 1
        assert list(observations["observations"]) == pytest.approx([value])

        start_date = smspec.start_date.to_datetime()
        time_index = smspec.keywords.index("TIME    ")
        days = smspec.units[time_index] == "DAYS    "
        restart_value = unsmry.steps[restart - 1].ministeps[-1].params[time_index]
        restart_time = start_date + (
            timedelta(days=float(restart_value))
            if days
            else timedelta(hours=float(restart_value))
        )

        assert abs(restart_time - observations["time"][0]) < timedelta(days=1.0)


@pytest.mark.usefixtures("use_tmpdir")
def test_that_summary_observations_can_use_restart_for_index_if_time_map_is_given():
    restart = 1
    time_map = ["2024-01-01", "2024-02-02"]

    obs_config_content = dedent(
        f"""
        SUMMARY_OBSERVATION FOPR_1
        {{
            KEY = FOPR;
            VALUE = 1;
            ERROR = 1;
            RESTART = {restart};
        }};
        """
    )
    Path("obs.conf").write_text(obs_config_content, encoding="utf-8")
    Path("time_map.txt").write_text("\n".join(time_map), encoding="utf-8")

    config_content = dedent(
        """
        NUM_REALIZATIONS 1
        ECLBASE ECLIPSE_CASE
        TIME_MAP time_map.txt
        OBS_CONFIG obs.conf
        """
    )
    config_file = Path("config.ert")
    config_file.write_text(config_content, encoding="utf-8")

    run_convert_observations(Namespace(config=str(config_file)))

    migrated_config = ErtConfig.from_file("config.ert")
    observations = create_observation_dataframes(
        migrated_config.observation_declarations, None
    )["summary"]

    # RESTART is a 1-based index; Python lists are 0-based.
    assert list(observations["time"]) == [datetime.fromisoformat(time_map[restart])]


def test_that_rft_config_is_created_from_observations():
    ert_config = ErtConfig.from_dict(
        {
            "ECLBASE": "ECLIPSE_CASE",
            "OBS_CONFIG": (
                "obsconf",
                [
                    {
                        "type": ObservationType.RFT,
                        "name": "NAME",
                        "WELL": "well",
                        "VALUE": "700",
                        "ERROR": "0.1",
                        "DATE": "2013-03-31",
                        "PROPERTY": "PRESSURE",
                        "NORTH": 71.0,
                        "EAST": 30.0,
                        "TVD": 2000,
                    }
                ],
            ),
        }
    )
    rft_config = cast(RFTConfig, ert_config.ensemble_config.response_configs["rft"])
    observations = create_observation_dataframes(
        ert_config.observation_declarations, rft_config
    )["rft"]
    assert_frame_equal(
        observations,
        pl.DataFrame(
            {
                "response_key": "well:2013-03-31:PRESSURE",
                "observation_key": "NAME",
                "east": pl.Series([30.0], dtype=pl.Float32),
                "north": pl.Series([71.0], dtype=pl.Float32),
                "tvd": pl.Series([2000.0], dtype=pl.Float32),
                "zone": pl.Series([None], dtype=pl.String),
                "observations": pl.Series([700.0], dtype=pl.Float32),
                "std": pl.Series([0.1], dtype=pl.Float32),
                "radius": pl.Series([None], dtype=pl.Float32),
            }
        ),
    )
    assert rft_config.data_to_read == {"well": {"2013-03-31": ["PRESSURE"]}}
    assert rft_config.locations == [(30.0, 71.0, 2000.0)]


def test_that_rft_observations_with_unknown_zones_errors():
    with pytest.raises(ConfigValidationError, match="no such zone"):
        ErtConfig.from_dict(
            {
                "ECLBASE": "ECLIPSE_CASE",
                "OBS_CONFIG": (
                    "obsconf",
                    [
                        {
                            "type": ObservationType.RFT,
                            "name": "NAME",
                            "WELL": "well",
                            "VALUE": "700",
                            "ERROR": "0.1",
                            "DATE": "2013-03-31",
                            "PROPERTY": "PRESSURE",
                            "NORTH": 71.0,
                            "EAST": 30.0,
                            "TVD": 2000,
                            "ZONE": "zone1",  # There is no such zone (no zonemap)
                        }
                    ],
                ),
            }
        )


@pytest.mark.usefixtures("use_tmpdir")
def test_that_the_date_keyword_sets_the_summary_index_without_time_map_or_refcase():
    date = "2020-01-01"

    # Write a legacy obs config using DATE for SUMMARY_OBSERVATION and
    # run the conversion to produce a migrated config on disk.
    obsconf = dedent(
        f"""
        SUMMARY_OBSERVATION FOPR_1 {{
            KEY = FOPR;
            VALUE = 1;
            ERROR = 1;
            DATE = {date};
        }};
        """
    )

    Path("obs.conf").write_text(obsconf, encoding="utf-8")

    config_content = dedent(
        """
        NUM_REALIZATIONS 1
        ECLBASE ECLIPSE_CASE
        OBS_CONFIG obs.conf
        """
    )
    Path("config.ert").write_text(config_content, encoding="utf-8")

    run_convert_observations(Namespace(config="config.ert"))

    migrated = ErtConfig.from_file("config.ert")
    observations = create_observation_dataframes(
        migrated.observation_declarations, None
    )["summary"]

    assert list(observations["time"]) == [datetime.fromisoformat(date)]


@given(
    st.integers(min_value=0, max_value=10000), st.floats(min_value=-1e9, max_value=1e9)
)
def test_that_general_observations_can_use_restart_even_without_refcase_and_time_map(
    restart, value
):
    ert_config: ErtConfig = ErtConfig.from_dict(
        {
            "GEN_DATA": [
                ["GEN", {"RESULT_FILE": "gen%d.txt", "REPORT_STEPS": str(restart)}]
            ],
            "OBS_CONFIG": (
                "obsconf",
                [
                    {
                        "type": ObservationType.GENERAL,
                        "name": "OBS",
                        "DATA": "GEN",
                        "RESTART": str(restart),
                        "VALUE": str(value),
                        "ERROR": "1.0",
                    }
                ],
            ),
        }
    )
    observations = create_observation_dataframes(
        observations=ert_config.observation_declarations,
        rft_config=None,
    )

    assert list(observations["gen_data"]["report_step"]) == [restart]
    assert list(observations["gen_data"]["observations"]) == pytest.approx([value])


@pytest.mark.usefixtures("use_tmpdir")
def test_that_the_date_keyword_sets_the_general_index_by_looking_up_time_map():
    restart = 1
    time_map = ["2024-01-01", "2024-02-02"]
    # Write a legacy obs config using DATE for GENERAL_OBSERVATION and
    # run the conversion to migrate DATE -> RESTART via the TIME_MAP.
    Path("time_map.txt").write_text("\n".join(time_map), encoding="utf-8")
    obsconf = dedent(f"""
    GENERAL_OBSERVATION OBS {{
        DATA = GEN;
        DATE = {time_map[restart]};
        VALUE = 1.0;
        ERROR = 1.0;
    }};
    """)

    Path("obsconf").write_text(obsconf, encoding="utf-8")

    config_content = dedent(
        """
        NUM_REALIZATIONS 1
        TIME_MAP time_map.txt
        GEN_DATA GEN RESULT_FILE:gen%d.txt REPORT_STEPS:1
        OBS_CONFIG obsconf
        """
    )
    Path("config.ert").write_text(config_content, encoding="utf-8")

    run_convert_observations(Namespace(config="config.ert"))
    observations = create_observation_dataframes(
        ErtConfig.from_file("config.ert").observation_declarations, None
    )
    assert observations["gen_data"].to_dicts()[0]["report_step"] == restart


@given(summary=summaries(), data=st.data())
@pytest.mark.integration_test
def test_that_the_date_keyword_sets_the_report_step_by_looking_up_refcase(
    tmp_path_factory: pytest.TempPathFactory, summary, data
):
    with pytest.MonkeyPatch.context() as patch:
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
        # Write a legacy obs config using DATE for GENERAL_OBSERVATION and
        # run the conversion to migrate DATE -> RESTART by looking up the REFCASE.
        obsconf = dedent(f"""
        GENERAL_OBSERVATION OBS {{
            DATA = GEN;
            DATE = {time_map[restart].isoformat()};
            VALUE = 1.0;
            ERROR = 1.0;
        }};
        """)

        Path("obsconf").write_text(obsconf, encoding="utf-8")

        config_content = dedent(
            f"""
            NUM_REALIZATIONS 1
            REFCASE ECLIPSE_CASE
            GEN_DATA GEN RESULT_FILE:gen%d.txt REPORT_STEPS:{restart}
            OBS_CONFIG obsconf
            """
        )
        Path("config.ert").write_text(config_content, encoding="utf-8")
        run_convert_observations(Namespace(config="config.ert"))
        observations = create_observation_dataframes(
            ErtConfig.from_file("config.ert").observation_declarations, None
        )
        assert observations["gen_data"].to_dicts()[0]["report_step"] == restart


@pytest.mark.parametrize("std", [-1.0, 0, 0.0])
def test_that_error_must_be_greater_than_zero_in_summary_observations(std):
    with pytest.raises(
        ConfigValidationError, match=r"must be given a strictly positive value"
    ):
        make_observations(
            "",
            [
                {
                    "type": ObservationType.SUMMARY,
                    "name": "FOPR",
                    "KEY": "FOPR",
                    "VALUE": "1",
                    "DATE": "2020-01-02",
                    "ERROR": str(std),
                }
            ],
        )


def test_that_computed_error_must_be_greater_than_zero_in_summary_observations():
    with pytest.raises(
        ConfigValidationError,
        match=r"must be given a strictly positive value",
    ):
        make_observations(
            "",
            [
                {
                    "type": ObservationType.SUMMARY,
                    "name": "FOPR",
                    "KEY": "FOPR",
                    "VALUE": "0",  # ERROR becomes zero when mode is REL
                    "DATE": "2020-01-02",
                    "ERROR": "1.0",
                    "ERROR_MODE": "REL",
                }
            ],
        )


@pytest.mark.usefixtures("use_tmpdir")
def test_that_absolute_error_must_be_greater_than_zero_in_history_observations():
    run_simulator()

    obsconf = dedent(
        """
        HISTORY_OBSERVATION FOPR {
            ERROR = 0.0;
            ERROR_MIN = 0.0;
        };
        """
    )

    Path("obs.conf").write_text(obsconf, encoding="utf-8")

    config_content = dedent(
        """
        NUM_REALIZATIONS 1
        ECLBASE BASEBASEBASE
        REFCASE MY_REFCASE
        SUMMARY *
        GEN_DATA GEN RESULT_FILE:gen%d.txt REPORT_STEPS:1
        OBS_CONFIG obs.conf
        """
    )
    Path("config.ert").write_text(config_content, encoding="utf-8")

    with pytest.raises(
        ConfigValidationError, match=r"must be given a strictly positive value"
    ):
        run_convert_observations(Namespace(config="config.ert"))


@pytest.mark.usefixtures("use_tmpdir")
def test_that_computed_error_must_be_greater_than_zero_in_history_observations():
    with pytest.raises(
        ConfigValidationError, match=r"must be given a strictly positive value"
    ):
        make_refcase_observations(
            """
                HISTORY_OBSERVATION FOPR {
                    ERROR=1.0;
                    ERROR_MODE=REL;
                };
            """,
            summary_values={
                "FOPR": FOPR_VALUE,
                "FOPRH": 0,  # ERROR becomes zero when mode is REL
            },
            parse=False,
        )


@pytest.mark.parametrize("std", [-1.0, 0, 0.0])
def test_that_error_must_be_greater_than_zero_in_general_observations(std):
    with pytest.raises(
        ConfigValidationError, match=r"must be given a strictly positive value"
    ):
        make_observations(
            "",
            [
                {
                    "type": ObservationType.GENERAL,
                    "name": "OBS",
                    "DATA": "GEN",
                    "RESTART": "1",
                    "INDEX_LIST": "1",
                    "VALUE": "1.0",
                    "ERROR": str(std),
                }
            ],
        )


def test_that_all_errors_in_general_observations_must_be_greater_than_zero(tmpdir):
    with tmpdir.as_cwd():
        # First error value will be 0
        Path("obs_data.txt").write_text(
            "\n".join(f"{float(i)} {float(i)}" for i in range(5)), encoding="utf-8"
        )
        with pytest.raises(
            ConfigValidationError, match=r"must be given a positive value|strictly > 0"
        ):
            make_observations(
                "",
                [
                    {
                        "type": ObservationType.GENERAL,
                        "name": "OBS",
                        "DATA": "GEN",
                        "RESTART": 1,
                        "OBS_FILE": "obs_data.txt",
                    }
                ],
            )


@pytest.mark.usefixtures("use_tmpdir")
def test_that_error_mode_is_not_allowed_in_general_observations():
    obsconf = dedent(
        """
        GENERAL_OBSERVATION OBS {
            DATA = GEN;
            DATE = 2020-01-02;
            INDEX_LIST = 1;
            VALUE = 1.0;
            ERROR = 0.1;
            ERROR_MODE = REL;
        };
        """
    )
    Path("obsconf").write_text(obsconf, encoding="utf-8")
    Path("config.ert").write_text(
        dedent(
            """
            NUM_REALIZATIONS 1
            GEN_DATA GEN RESULT_FILE:gen%d.txt REPORT_STEPS:1
            OBS_CONFIG obsconf
            """
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigValidationError, match=r"Unknown ERROR_MODE"):
        run_convert_observations(Namespace(config="config.ert"))


@pytest.mark.usefixtures("use_tmpdir")
def test_that_error_min_is_not_allowed_in_general_observations():
    obsconf = dedent(
        """
        GENERAL_OBSERVATION OBS {
            DATA = GEN;
            DATE = 2020-01-02;
            INDEX_LIST = 1;
            VALUE = 1.0;
            ERROR = 0.1;
            ERROR_MIN = 0.05;
        };
        """
    )
    Path("obsconf").write_text(obsconf, encoding="utf-8")
    Path("config.ert").write_text(
        dedent(
            """
            NUM_REALIZATIONS 1
            GEN_DATA GEN RESULT_FILE:gen%d.txt REPORT_STEPS:1
            OBS_CONFIG obsconf
            """
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigValidationError, match=r"Unknown ERROR_MIN"):
        run_convert_observations(Namespace(config="config.ert"))


def test_that_empty_observations_file_causes_exception():
    with pytest.raises(
        expected_exception=ConfigValidationError,
        match="Empty observations file",
    ):
        ErtConfig.from_dict({"OBS_CONFIG": ("obs_conf", "")})


@pytest.mark.usefixtures("use_tmpdir")
def test_that_having_no_refcase_but_history_observations_causes_exception():
    Path("obs.conf").write_text("HISTORY_OBSERVATION FOPR;", encoding="utf-8")

    config_content = dedent(
        """
        NUM_REALIZATIONS 1
        ECLBASE my_case
        OBS_CONFIG obs.conf
        """
    )
    Path("config.ert").write_text(config_content, encoding="utf-8")

    with pytest.raises(
        ConfigValidationError,
        match="REFCASE is required for HISTORY_OBSERVATION",
    ):
        run_convert_observations(Namespace(config="config.ert"))


def test_that_index_list_is_read(tmpdir):
    with tmpdir.as_cwd():
        Path("obs_data.txt").write_text(
            "\n".join(f"{float(i)} 0.1" for i in range(5)), encoding="utf-8"
        )
        obs = make_observations(
            "",
            [
                {
                    "type": ObservationType.GENERAL,
                    "name": "OBS",
                    "DATA": "GEN",
                    "INDEX_LIST": "0,2,4,6,8",
                    "RESTART": "1",
                    "OBS_FILE": "obs_data.txt",
                }
            ],
        )
        observations = create_observation_dataframes(
            observations=obs,
            rft_config=None,
        )
        assert list(observations["gen_data"]["index"]) == [0, 2, 4, 6, 8]


@pytest.mark.usefixtures("use_tmpdir")
def test_that_invalid_time_map_file_raises_config_validation_error():
    Path("time_map.txt").write_text("invalid content", encoding="utf-8")
    Path("obs.conf").write_text(
        dedent(
            """
        GENERAL_OBSERVATION GEN_OBS {
            DATA = GEN_DATA_KEY;
            OBS_FILE = obs_data.txt;
            DATE = 2023-01-01;
        };
        """
        ),
        encoding="utf-8",
    )
    Path("obs_data.txt").write_text("1.0", encoding="utf-8")
    Path("config.ert").write_text(
        dedent(
            """
            NUM_REALIZATIONS 1
            TIME_MAP time_map.txt
            OBS_CONFIG obs.conf
            """
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigValidationError, match="Could not read timemap file"):
        run_convert_observations(Namespace(config="config.ert"))


def test_that_index_file_is_read(tmpdir):
    with tmpdir.as_cwd():
        Path("obs_idx.txt").write_text("0\n2\n4\n6\n8", encoding="utf-8")
        Path("obs_data.txt").write_text(
            "\n".join(f"{float(i)} 0.1\n" for i in range(5)), encoding="utf-8"
        )
        obs = make_observations(
            "",
            [
                {
                    "type": ObservationType.GENERAL,
                    "name": "OBS",
                    "DATA": "GEN",
                    "RESTART": "1",
                    "INDEX_FILE": "obs_idx.txt",
                    "OBS_FILE": "obs_data.txt",
                }
            ],
        )
        observations = create_observation_dataframes(
            observations=obs,
            rft_config=None,
        )
        assert list(observations["gen_data"]["index"]) == [0, 2, 4, 6, 8]


def test_that_non_existent_obs_file_is_invalid():
    with pytest.raises(
        expected_exception=ConfigValidationError,
        match="did not resolve to a valid path",
    ):
        make_observations(
            "",
            [
                {
                    "type": ObservationType.GENERAL,
                    "name": "OBS",
                    "DATA": "RES",
                    "INDEX_LIST": "0,2,4,6,8",
                    "RESTART": "0",
                    "OBS_FILE": "does_not_exist/at_all",
                }
            ],
        )


@pytest.mark.usefixtures("use_tmpdir")
def test_that_non_existent_time_map_file_is_invalid():
    obsconf = dedent(
        """
        GENERAL_OBSERVATION OBS {
            DATA = RES;
            INDEX_LIST = 0;
            DATE = 2017-11-09;
            VALUE = 0.0;
            ERROR = 0.0;
        };
        """
    )

    Path("obsconf").write_text(obsconf, encoding="utf-8")
    config_content = dedent(
        """
        NUM_REALIZATIONS 1
        GEN_DATA RES RESULT_FILE:out
        OBS_CONFIG obsconf
        """
    )
    Path("config.ert").write_text(config_content, encoding="utf-8")

    with pytest.raises(ObservationConfigError, match="TIME_MAP"):
        run_convert_observations(Namespace(config="config.ert"))


@pytest.mark.usefixtures("use_tmpdir")
def test_that_general_observation_cannot_contain_both_value_and_obs_file():
    Path("obs_idx.txt").write_text("0\n2\n4\n6\n8", encoding="utf-8")
    Path("obs_data.txt").write_text(
        "\n".join(f"{float(i)} 0.1" for i in range(5)), encoding="utf-8"
    )
    with pytest.raises(
        ConfigValidationError, match=r"cannot contain both VALUE.*OBS_FILE"
    ):
        make_observations(
            "",
            [
                {
                    "type": ObservationType.GENERAL,
                    "name": "OBS",
                    "DATA": "GEN",
                    "RESTART": "1",
                    "INDEX_FILE": "obs_idx.txt",
                    "OBS_FILE": "obs_data.txt",
                    "VALUE": "1.0",
                    "ERROR": "0.1",
                }
            ],
        )


def test_that_general_observation_must_contain_either_value_or_obs_file():
    with pytest.raises(
        ConfigValidationError, match=r"must contain either VALUE.*OBS_FILE"
    ):
        make_observations(
            "",
            [
                {
                    "type": ObservationType.GENERAL,
                    "name": "OBS",
                    "DATA": "GEN",
                    "RESTART": "1",
                }
            ],
        )


def test_that_non_numbers_in_obs_file_shows_informative_error_message(tmpdir):
    with tmpdir.as_cwd():
        Path("obs_data.txt").write_text("not_an_int 0.1\n", encoding="utf-8")
        with pytest.raises(
            expected_exception=ConfigValidationError,
            match=r"Failed to read OBS_FILE obs_data.txt: could not convert"
            " string 'not_an_int' to float64 at row 0, column 1",
        ):
            make_observations(
                "",
                [
                    {
                        "type": ObservationType.GENERAL,
                        "name": "OBS",
                        "DATA": "GEN",
                        "INDEX_LIST": "0,2,4,6,8",
                        "RESTART": 1,
                        "OBS_FILE": "obs_data.txt",
                    }
                ],
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
            "",
            [
                {
                    "type": ObservationType.GENERAL,
                    "name": "OBS",
                    "DATA": "GEN",
                    "INDEX_LIST": "0,2,4,6,8",
                    "RESTART": 1,
                    "OBS_FILE": "obs_data.txt",
                }
            ],
        )


@pytest.mark.usefixtures("use_tmpdir")
def test_that_the_number_of_values_in_obs_file_must_be_even():
    with open("obs_data.txt", "w", encoding="utf-8") as fh:
        fh.writelines(f"{float(i)} 0.1 0.1\n" for i in range(5))
    with pytest.raises(ConfigValidationError, match="Expected even number of values"):
        make_observations(
            "",
            [
                {
                    "type": ObservationType.GENERAL,
                    "name": "OBS",
                    "DATA": "GEN",
                    "INDEX_LIST": "0,2,4,6,8",
                    "RESTART": "1",
                    "OBS_FILE": "obs_data.txt",
                }
            ],
        )


def test_that_giving_both_index_file_and_index_list_raises_an_exception(tmpdir):
    with tmpdir.as_cwd():
        Path("obs_idx.txt").write_text("0\n2\n4\n6\n8", encoding="utf-8")
        with pytest.raises(
            expected_exception=ConfigValidationError,
            match="both INDEX_FILE and INDEX_LIST",
        ):
            make_observations(
                "",
                [
                    {
                        "type": ObservationType.GENERAL,
                        "name": "OBS",
                        "DATA": "GEN",
                        "INDEX_LIST": "0,2,4,6,8",
                        "INDEX_FILE": "obs_idx.txt",
                        "RESTART": "1",
                        "VALUE": "0.0",
                        "ERROR": "0.1",
                    }
                ],
            )


def run_sim(start_date, keys=None, values=None, days=None):
    """Create :term:`summary files`"""
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
    ("time_map_statement", "time_map_creator"),
    [
        ({"REFCASE": "ECLIPSE_CASE"}, lambda: run_sim(datetime(2014, 9, 10))),
        (
            {"TIME_MAP": ("time_map.txt", "2014-09-10\n2014-09-11\n")},
            lambda: None,
        ),
    ],
)
@pytest.mark.parametrize(
    ("time_unit", "time_delta", "expectation"),
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
                match=r".*Could not find 2014-09-12 00:00:00 \(DAYS=2.0\)"
                " in the time map for observations FOPR_1",
            ),
            id="Outside tolerance days",
        ),
        pytest.param("HOURS", 24.0, does_not_raise(), id="1 day in hours"),
        # pytest.param( # Not migrated; no refcase/time_map provided.
        #    "HOURS",
        #    48.0,
        #    pytest.raises(
        #        ConfigValidationError,
        #        match=r".*Could not find 2014-09-12 00:00:00 \(HOURS=48.0\)"
        #        " in the time map for observations FOPR_1",
        #    ),
        #    id="Outside tolerance hours",
        # ),
        pytest.param("DATE", "2014-09-11", does_not_raise(), id="1 day in date"),
        # pytest.param( # Not migrated; no refcase/time_map provided.
        #    "DATE",
        #    "2014-09-12",
        #    pytest.raises(
        #        ConfigValidationError,
        #        match=r".*Could not find 2014-09-12 00:00:00 \(DATE=2014-09-12\)"
        #        " in the time map for observations FOPR_1",
        #    ),
        #    id="Outside tolerance in date",
        # ),
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
        if time_map_creator:
            time_map_creator()

        obs_config_content = dedent(
            f"""
        SUMMARY_OBSERVATION FOPR_1
        {{
            VALUE   = 0.1;
            ERROR   = 0.05;
            {time_unit}    = {time_delta};
            KEY     = FOPR;
        }};
        """
        )
        Path("obsconf").write_text(obs_config_content, encoding="utf-8")

        config_lines = [
            "NUM_REALIZATIONS 1",
            "ECLBASE ECLIPSE_CASE",
            "OBS_CONFIG obsconf",
        ]
        if "TIME_MAP" in time_map_statement:
            time_map_file, time_map_data = time_map_statement["TIME_MAP"]
            Path(time_map_file).write_text(time_map_data, encoding="utf-8")
            config_lines.append(f"TIME_MAP {time_map_file}")
        elif "REFCASE" in time_map_statement:
            config_lines.append(f"REFCASE {time_map_statement['REFCASE']}")
        Path("config.ert").write_text("\n".join(config_lines), encoding="utf-8")

        with expectation:
            run_convert_observations(Namespace(config="config.ert"))
            ErtConfig.from_file("config.ert")


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.filterwarnings(
    r"ignore:.*Segment [^\s]+ "
    "((does not contain any time steps)|(out of bounds)|(start after stop)).*"
    ":ert.config.ConfigWarning"
)
@pytest.mark.parametrize(
    ("start", "stop", "message"),
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
    ],
)
def test_that_out_of_bounds_segments_are_truncated(tmpdir, start, stop, message):
    with tmpdir.as_cwd():
        run_sim(
            datetime(2014, 9, 10),
            [("FOPR", "SM3/DAY", None), ("FOPRH", "SM3/DAY", None)],
        )

        obsconf = dedent(f"""
        HISTORY_OBSERVATION FOPR {{
            ERROR = 0.20;
            ERROR_MODE = RELMIN;
            ERROR_MIN = 100;
            SEGMENT FIRST_YEAR {{
                START = {start};
                STOP  = {stop};
                ERROR = 0.50;
                ERROR_MODE = REL;
            }};
        }};
        """)

        Path("obsconf").write_text(obsconf, encoding="utf-8")

        config_content = dedent(
            """
            NUM_REALIZATIONS 1
            ECLBASE ECLIPSE_CASE
            REFCASE ECLIPSE_CASE
            OBS_CONFIG obsconf
            """
        )
        Path("config.ert").write_text(config_content, encoding="utf-8")

        with pytest.warns(ConfigWarning, match=message):
            run_convert_observations(Namespace(config="config.ert"))


@given(
    std=st.floats(min_value=0.1, max_value=1.0e3),
    with_ext=st.booleans(),
    summary=summaries(summary_keys=st.just(["FOPR", "FOPRH"])),
)
def test_that_history_observations_values_are_fetched_from_refcase(
    tmp_path_factory: pytest.TempPathFactory, summary, with_ext, std
):
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(tmp_path_factory.mktemp("history_observation_values_are_fetched"))
        smspec, unsmry = summary
        smspec.to_file("ECLIPSE_CASE.SMSPEC")
        unsmry.to_file("ECLIPSE_CASE.UNSMRY")

        obsconf = dedent(
            f"""
            HISTORY_OBSERVATION FOPR {{
                ERROR = {std};
                ERROR_MODE = ABS;
            }};
            """
        )

        Path("obsconf").write_text(obsconf, encoding="utf-8")

        config_content = dedent(
            f"""
            NUM_REALIZATIONS 1
            ECLBASE ECLIPSE_CASE
            REFCASE {f"ECLIPSE_CASE{'.DATA'}" if with_ext else "ECLIPSE_CASE"}
            OBS_CONFIG obsconf
            """
        )
        Path("config.ert").write_text(config_content, encoding="utf-8")

        run_convert_observations(Namespace(config="config.ert"))
        observations = create_observation_dataframes(
            ErtConfig.from_file("config.ert").observation_declarations, None
        )["summary"]

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
    Path("obs_idx.txt").write_text("0\n2\n4\n6", encoding="utf-8")
    Path("obs_data.txt").write_text(
        "\n".join(f"{float(i)} 0.1" for i in range(5)), encoding="utf-8"
    )

    with pytest.raises(ConfigValidationError, match="must be of equal length"):
        ErtConfig.from_dict(
            {
                "NUM_REALIZATIONS": 2,
                "GEN_DATA": [["RES", {"RESULT_FILE": "out"}]],
                "OBS_CONFIG": (
                    "obsconf",
                    [
                        {
                            "type": ObservationType.GENERAL,
                            "name": "OBS",
                            "DATA": "RES",
                            "INDEX_FILE": "obs_idx.txt",  # shorter than obs_file
                            "OBS_FILE": "obs_data.txt",
                        }
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
                            {
                                "type": ObservationType.GENERAL,
                                "name": "OBS",
                                "DATA": "RES",
                                "INDEX_LIST": "200",  # shorter than obs_file
                                "OBS_FILE": "obs_data.txt",
                            }
                        ],
                    ),
                }
            )


@pytest.mark.usefixtures("use_tmpdir")
def test_that_general_observations_data_must_match_a_gen_datas_name():
    with pytest.raises(
        ConfigValidationError,
        match="No GEN_DATA with name 'RES' found",
    ):
        ErtConfig.from_dict(
            {
                "NUM_REALIZATIONS": 2,
                "GEN_DATA": [["OTHER", {"RESULT_FILE": "out"}]],
                "OBS_CONFIG": (
                    "obsconf",
                    [
                        {
                            "type": ObservationType.GENERAL,
                            "name": "OBS",
                            "DATA": "RES",
                            "INDEX_LIST": "0,2,4,6,8",
                            "RESTART": "0",
                            "VALUE": "1",
                            "ERROR": "1",
                        }
                    ],
                ),
            }
        )


@pytest.mark.usefixtures("use_tmpdir")
def test_that_general_observation_restart_must_match_gen_data_report_step():
    with pytest.raises(
        ConfigValidationError,
        match="is not configured to load from report step",
    ):
        ErtConfig.from_dict(
            {
                "NUM_REALIZATIONS": 2,
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
                        {
                            "type": ObservationType.GENERAL,
                            "name": "OBS",
                            "DATA": "RES",
                            "INDEX_LIST": "0,2,4,6,8",
                            "RESTART": "0",
                            "VALUE": "1",
                            "ERROR": "1",
                        }
                    ],
                ),
            }
        )


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

        obsconf = dedent(
            """
            HISTORY_OBSERVATION FOPR {
                ERROR = 0.20;
                ERROR_MODE = ABS;
            };
            HISTORY_OBSERVATION FGPR {
                ERROR = 0.1;
                ERROR_MODE = REL;
            };
            HISTORY_OBSERVATION FWPR {
                ERROR = 0.1;
                ERROR_MODE = RELMIN;
                ERROR_MIN = 10000;
            };
            """
        )

        Path("obsconf").write_text(obsconf, encoding="utf-8")

        config_content = dedent(
            """
            NUM_REALIZATIONS 1
            ECLBASE ECLIPSE_CASE
            REFCASE ECLIPSE_CASE
            OBS_CONFIG obsconf
            """
        )
        Path("config.ert").write_text(config_content, encoding="utf-8")

        run_convert_observations(Namespace(config="config.ert"))
        observations = create_observation_dataframes(
            ErtConfig.from_file("config.ert").observation_declarations, None
        )["summary"]

        assert list(observations["response_key"]) == ["FGPR", "FOPR", "FWPR"]
        assert list(observations["observations"]) == pytest.approx([15, 20, 25])
        assert list(observations["std"]) == pytest.approx([1.5, 0.2, 10000])


def test_that_segment_defaults_are_applied(tmpdir):
    with tmpdir.as_cwd():
        run_sim(
            datetime(2014, 9, 10),
            [("FOPR", "SM3/DAY", None), ("FOPRH", "SM3/DAY", None)],
            days=range(10),
        )

        obsconf = dedent(
            """
            HISTORY_OBSERVATION FOPR {
                ERROR = 1.0;
                SEGMENT SEG {
                    START = 5;
                    STOP  = 10;
                    ERROR = 0.05;
                };
            };
            """
        )

        Path("obsconf").write_text(obsconf, encoding="utf-8")

        config_content = dedent(
            """
            NUM_REALIZATIONS 1
            ECLBASE ECLIPSE_CASE
            REFCASE ECLIPSE_CASE
            OBS_CONFIG obsconf
            """
        )
        Path("config.ert").write_text(config_content, encoding="utf-8")

        run_convert_observations(Namespace(config="config.ert"))
        observations = create_observation_dataframes(
            ErtConfig.from_file("config.ert").observation_declarations, None
        )["summary"]

        # default error_min is 0.1
        # default error method is RELMIN
        # default error is 0.1
        assert list(observations["std"]) == pytest.approx([1.0] * 5 + [0.1] * 5)


def test_that_summary_default_error_min_is_applied():
    obs = make_observations(
        "",
        [
            {
                "type": ObservationType.SUMMARY,
                "name": "FOPR",
                "VALUE": "1",
                "ERROR": "0.01",
                "KEY": "FOPR",
                "DATE": "2020-01-02",
                "ERROR_MODE": "RELMIN",
            }
        ],
    )
    observations = create_observation_dataframes(
        obs,
        rft_config=None,
    )

    # default error_min is 0.1
    assert list(observations["summary"]["std"]) == pytest.approx([0.1])


@pytest.mark.usefixtures("use_tmpdir")
def test_that_start_must_be_set_in_a_segment():
    obsconf = dedent(
        """
        HISTORY_OBSERVATION  FOPR {
           ERROR      = 0.1;

           SEGMENT SEG
           {
              STOP  = 1;
              ERROR = 0.50;
           };
        };
        """
    )
    Path("obsconf").write_text(obsconf, encoding="utf-8")
    Path("config.ert").write_text(
        dedent(
            """
            NUM_REALIZATIONS 1
            ECLBASE ECLIPSE_CASE
            OBS_CONFIG obsconf
            """
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigValidationError, match='Missing item "START"'):
        run_convert_observations(Namespace(config="config.ert"))


@pytest.mark.usefixtures("use_tmpdir")
def test_that_stop_must_be_set_in_a_segment():
    obsconf = dedent(
        """
        HISTORY_OBSERVATION FOPR {
           ERROR      = 0.1;

           SEGMENT SEG {
              START  = 1;
              ERROR = 0.50;
           };
        };
        """
    )
    Path("obsconf").write_text(obsconf, encoding="utf-8")
    Path("config.ert").write_text(
        dedent(
            """
            NUM_REALIZATIONS 1
            ECLBASE ECLIPSE_CASE
            OBS_CONFIG obsconf
            """
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigValidationError, match='Missing item "STOP"'):
        run_convert_observations(Namespace(config="config.ert"))


@pytest.mark.usefixtures("use_tmpdir")
def test_that_stop_must_be_given_integer_value():
    obsconf = dedent(
        """
        HISTORY_OBSERVATION FOPR {
           ERROR      = 0.1;

           SEGMENT SEG
           {
              START = 0;
              STOP  = 3.2;
              ERROR = 0.50;
           };
        };
    """
    )
    Path("obsconf").write_text(obsconf, encoding="utf-8")
    Path("config.ert").write_text(
        dedent(
            """
            NUM_REALIZATIONS 1
            ECLBASE ECLIPSE_CASE
            OBS_CONFIG obsconf
            """
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigValidationError, match=r'Failed to validate "3\.2"'):
        run_convert_observations(Namespace(config="config.ert"))


@pytest.mark.usefixtures("use_tmpdir")
def test_that_start_must_be_given_integer_value():
    obsconf = dedent(
        """
        HISTORY_OBSERVATION FOPR {
           ERROR      = 0.1;

           SEGMENT SEG
           {
              START = 1.1;
              STOP  = 0;
              ERROR = 0.50;
           };
        };
    """
    )
    Path("obsconf").write_text(obsconf, encoding="utf-8")
    Path("config.ert").write_text(
        dedent(
            """
            NUM_REALIZATIONS 1
            ECLBASE ECLIPSE_CASE
            OBS_CONFIG obsconf
            """
        ),
        encoding="utf-8",
    )
    with pytest.raises(ObservationConfigError, match=r"Failed to validate"):
        run_convert_observations(Namespace(config="config.ert"))


@pytest.mark.parametrize(
    "segment_property",
    ["ERROR", "ERROR_MIN"],
)
@pytest.mark.usefixtures("use_tmpdir")
def test_that_property_must_be_positive_in_a_segment(segment_property):
    obsconf = dedent(
        f"""
        HISTORY_OBSERVATION FOPR {{
           ERROR      = 0.1;
           SEGMENT SEG {{
              START = 1;
              STOP  = 0;
              {segment_property} = -1;
           }};
        }};
    """
    )
    Path("obsconf").write_text(obsconf, encoding="utf-8")
    Path("config.ert").write_text(
        dedent(
            """
            NUM_REALIZATIONS 1
            ECLBASE ECLIPSE_CASE
            OBS_CONFIG obsconf
            """
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigValidationError, match='Failed to validate "-1"'):
        run_convert_observations(Namespace(config="config.ert"))


@pytest.mark.usefixtures("use_tmpdir")
def test_that_error_mode_must_be_one_of_rel_abs_relmin_in_a_segment():
    obsconf = dedent(
        """
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
    """
    )
    Path("obsconf").write_text(obsconf, encoding="utf-8")
    Path("config.ert").write_text(
        dedent(
            """
            NUM_REALIZATIONS 1
            ECLBASE ECLIPSE_CASE
            OBS_CONFIG obsconf
            """
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigValidationError, match='Failed to validate "NOT_ABS"'):
        run_convert_observations(Namespace(config="config.ert"))


@pytest.mark.usefixtures("use_tmpdir")
def test_that_restart_must_be_positive_in_a_summary_observation():
    obsconf = dedent(
        """
        SUMMARY_OBSERVATION FOPR {
            RESTART = -1;
            KEY = FOPR;
            VALUE = 1.0;
            ERROR = 0.1;
        };
        """
    )
    Path("obs.conf").write_text(obsconf, encoding="utf-8")
    Path("time_map.txt").write_text("2020-01-01\n2020-01-02\n", encoding="utf-8")
    config_content = dedent(
        """
        NUM_REALIZATIONS 1
        TIME_MAP time_map.txt
        OBS_CONFIG obs.conf
        """
    )
    Path("config.ert").write_text(config_content, encoding="utf-8")

    with pytest.raises(ConfigValidationError, match='Failed to validate "-1"'):
        run_convert_observations(Namespace(config="config.ert"))


@pytest.mark.usefixtures("use_tmpdir")
def test_that_restart_must_be_a_number_in_summary_observation():
    obsconf = dedent(
        """
        SUMMARY_OBSERVATION FOPR {
            RESTART = minus_one;
            KEY = FOPR;
            VALUE = 1.0;
            ERROR = 0.1;
        };
        """
    )
    Path("obs.conf").write_text(obsconf, encoding="utf-8")
    Path("time_map.txt").write_text("2020-01-01\n2020-01-02\n", encoding="utf-8")
    config_content = dedent(
        """
        NUM_REALIZATIONS 1
        TIME_MAP time_map.txt
        OBS_CONFIG obs.conf
        """
    )
    Path("config.ert").write_text(config_content, encoding="utf-8")

    with pytest.raises(ConfigValidationError, match='Failed to validate "minus_one"'):
        run_convert_observations(Namespace(config="config.ert"))


def test_that_value_must_be_set_in_summary_observation():
    with pytest.raises(ConfigValidationError, match='Missing item "VALUE"'):
        ert_config_from_parser("SUMMARY_OBSERVATION FOPR {DATE = 2025-01-01;};")


def test_that_key_must_be_set_in_summary_observation():
    with pytest.raises(ConfigValidationError, match='Missing item "KEY"'):
        ert_config_from_parser("""
            SUMMARY_OBSERVATION  FOPR {
               VALUE = 1;
               ERROR = 0.1;
            };
        """)


def test_that_data_must_be_set_in_general_observation():
    with pytest.raises(ConfigValidationError, match='Missing item "DATA"'):
        ert_config_from_parser("""
            GENERAL_OBSERVATION obs {
               DATE       = 2023-02-01;
               VALUE      = 1;
               ERROR      = 0.01;
               ERROR_MIN  = 0.1;
            };
        """)


@pytest.mark.usefixtures("use_tmpdir")
def test_that_error_must_be_a_positive_number_in_history_observation():
    obsconf = "HISTORY_OBSERVATION FOPR { ERROR = -1;};"
    Path("obsconf").write_text(obsconf, encoding="utf-8")
    Path("config.ert").write_text(
        dedent(
            """
            NUM_REALIZATIONS 1
            ECLBASE BASEBASEBASE
            OBS_CONFIG obsconf
            """
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigValidationError, match='Failed to validate "-1"'):
        run_convert_observations(Namespace(config="config.ert"))


@pytest.mark.usefixtures("use_tmpdir")
def test_that_error_min_must_be_a_positive_number_in_history_observation():
    obsconf = dedent(
        """
        HISTORY_OBSERVATION FOPR {
            ERROR_MODE=RELMIN;
            ERROR_MIN = -1;
            ERROR=1.0;
        };
        """
    )
    Path("obsconf").write_text(obsconf, encoding="utf-8")
    Path("config.ert").write_text(
        dedent(
            """
            NUM_REALIZATIONS 1
            ECLBASE BASEBASEBASE
            OBS_CONFIG obsconf
            """
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigValidationError, match='Failed to validate "-1"'):
        run_convert_observations(Namespace(config="config.ert"))


@pytest.mark.usefixtures("use_tmpdir")
def test_that_error_mode_must_be_one_of_rel_abs_relmin_in_history_observation():
    obsconf = dedent(
        """
        HISTORY_OBSERVATION  FOPR {
            ERROR_MODE = NOT_ABS;
            ERROR=1.0;
        };
        """
    )
    Path("obsconf").write_text(obsconf, encoding="utf-8")
    Path("config.ert").write_text(
        dedent(
            """
            NUM_REALIZATIONS 1
            ECLBASE BASEBASEBASE
            OBS_CONFIG obsconf
            """
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigValidationError, match='Failed to validate "NOT_ABS"'):
        run_convert_observations(Namespace(config="config.ert"))


@pytest.mark.usefixtures("use_tmpdir")
def test_that_history_observations_can_omit_body():
    obs = make_refcase_observations("HISTORY_OBSERVATION  FOPR;")
    assert list(obs["summary"]["response_key"]) == ["FOPR"]


def test_that_error_min_must_be_a_positive_number_in_summary_observation():
    with pytest.raises(ConfigValidationError, match='Failed to validate "-1"'):
        ert_config_from_parser("""
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
        ert_config_from_parser("""
            SUMMARY_OBSERVATION  FOPR
            {
                ERROR_MODE = NOT_ABS;
                ERROR=1.0;
                RESTART = 1;
                VALUE=1.0;
                KEY = FOPR;
            };
        """)


@pytest.mark.parametrize(
    "general_property",
    ["DAYS", "HOURS"],
)
@pytest.mark.usefixtures("use_tmpdir")
def test_that_property_must_be_a_positive_number_in_general_observation(
    general_property,
):
    obsconf = dedent(
        f"""
        GENERAL_OBSERVATION FOPR {{
            {general_property} = -1;
            DATA = GEN;
        }};
        """
    )
    Path("obsconf").write_text(obsconf, encoding="utf-8")
    Path("config.ert").write_text(
        dedent(
            """
            NUM_REALIZATIONS 1
            GEN_DATA GEN RESULT_FILE:gen%d.txt REPORT_STEPS:1
            OBS_CONFIG obsconf
            """
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigValidationError, match='Failed to validate "-1"'):
        run_convert_observations(Namespace(config="config.ert"))


@pytest.mark.usefixtures("use_tmpdir")
def test_that_date_must_be_a_date_in_general_observation():
    obsconf = dedent(
        """
        GENERAL_OBSERVATION FOPR
        {
            DATE = wednesday;
            VALUE = 1.0;
            ERROR = 0.1;
            DATA = GEN;
        };
        """
    )

    Path("obsconf").write_text(obsconf, encoding="utf-8")
    Path("time_map.txt").write_text("2014-09-10\n2014-09-11\n", encoding="utf-8")
    Path("config.ert").write_text(
        dedent(
            """
            NUM_REALIZATIONS 1
            ECLBASE ECLIPSE_CASE
            TIME_MAP time_map.txt
            GEN_DATA GEN RESULT_FILE:gen%d.txt REPORT_STEPS:1
            OBS_CONFIG obsconf
            """
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigValidationError, match="Please use ISO date format"):
        run_convert_observations(Namespace(config="config.ert"))


def test_that_value_must_be_a_number_in_general_observation():
    with pytest.raises(ConfigValidationError, match='Failed to validate "exactly_1"'):
        ert_config_from_parser("""
            GENERAL_OBSERVATION FOPR
            {
                ERROR = 0;
                VALUE = exactly_1;
                DATA = GEN;
            };
        """)


def test_that_error_must_be_set_in_general_observation():
    with pytest.raises(ConfigValidationError, match="ERROR"):
        ert_config_from_parser("""
            GENERAL_OBSERVATION FOPR
            {
                VALUE = 1;
                DATA = GEN;
            };
        """)


@pytest.mark.parametrize(
    "summary_property",
    ["DAYS", "HOURS"],
)
@pytest.mark.usefixtures("use_tmpdir")
def test_that_property_must_be_a_positive_number_in_summary_observation(
    summary_property,
):
    obsconf = dedent(
        f"""
        SUMMARY_OBSERVATION FOPR {{
            {summary_property} = -1;
            KEY = FOPR;
        }};
        """
    )
    Path("obs.conf").write_text(obsconf, encoding="utf-8")
    Path("config.ert").write_text(
        dedent(
            """
            NUM_REALIZATIONS 1
            ECLBASE ECLIPSE_CASE
            OBS_CONFIG obs.conf
            """
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigValidationError, match='Failed to validate "-1"'):
        run_convert_observations(Namespace(config="config.ert"))


def test_that_date_must_be_a_date_in_summary_observation():
    with pytest.raises(ConfigValidationError, match="Please use ISO date format"):
        ert_config_from_parser("""
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
        ert_config_from_parser("""
            SUMMARY_OBSERVATION FOPR
            {
                VALUE = exactly_1;
                KEY = FOPR;
            };
        """)


def test_that_error_must_be_set_in_summary_observation():
    with pytest.raises(ConfigValidationError, match="ERROR"):
        ert_config_from_parser("""
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
@pytest.mark.parametrize("unknown_key", ["SMERROR", "name", "type", "segments"])
@pytest.mark.usefixtures("use_tmpdir")
def test_that_setting_an_unknown_key_is_not_valid(observation_type, unknown_key):
    if observation_type == "HISTORY_OBSERVATION":
        # HISTORY_OBSERVATION is deprecated; write legacy obs file and run migration
        obsconf = f"{observation_type} FOPR {{{unknown_key}=0.1;DATA=key;}};"
        Path("obsconf").write_text(obsconf, encoding="utf-8")
        Path("config.ert").write_text(
            dedent(
                """
                NUM_REALIZATIONS 1
                ECLBASE ECLIPSE_CASE
                OBS_CONFIG obsconf
                """
            ),
            encoding="utf-8",
        )
        with pytest.raises(ConfigValidationError, match=f"Unknown {unknown_key}"):
            run_convert_observations(Namespace(config="config.ert"))
    else:
        with pytest.raises(ConfigValidationError, match=f"Unknown {unknown_key}"):
            ert_config_from_parser(
                f"{observation_type} FOPR {{{unknown_key}=0.1;DATA=key;}};"
            )


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize("unknown_key", ["SMERROR", "name", "type", "segments"])
def test_that_setting_an_unknown_key_in_a_segment_is_not_valid(unknown_key):
    obsconf = dedent(
        f"""
        HISTORY_OBSERVATION FOPR {{
            SEGMENT FIRST_YEAR {{
                START = 1;
                STOP = 2;
                {unknown_key} = 0.02;
            }};
        }};
        """
    )
    Path("obsconf").write_text(obsconf, encoding="utf-8")
    Path("config.ert").write_text(
        dedent(
            """
            NUM_REALIZATIONS 1
            ECLBASE ECLIPSE_CASE
            OBS_CONFIG obsconf
            """
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigValidationError, match=f"Unknown {unknown_key}"):
        run_convert_observations(Namespace(config="config.ert"))


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.filterwarnings(
    r"ignore:.*((does not contain any time steps)|(out of bounds)|(start after stop)).*"
)
def test_ert_config_logs_observation_types_and_keywords(caplog):
    obs_config_contents = """
    GENERAL_OBSERVATION OBS1 {
        VALUE = 1;
        DATA = GEN;
        ERROR = 0.1;
        RESTART = 1;
    };
    HISTORY_OBSERVATION FWPR;
    HISTORY_OBSERVATION FOPR {
        SEGMENT FIRST_YEAR {
            START = 1;
            STOP = 2;
            ERROR = 0.02;
        };
    };
    SUMMARY_OBSERVATION SUMOP {
        VALUE = 1;
        ERROR = 0.1;
        KEY = FGPR;
        RESTART = 1;
    };
    """
    with caplog.at_level(logging.INFO):
        make_refcase_observations(
            obs_config_contents,
            summary_values={"FOPR": 1, "FOPRH": 2, "FWPR": 3, "FWPRH": 4},
        )
    assert "Count of observation types" in caplog.text
    assert "GENERAL_OBSERVATION" in caplog.text
    # HISTORY observations are converted; ensure migration trace is present
    assert "History obs" in caplog.text
    assert "SUMMARY_OBSERVATION" in caplog.text
    assert "Count of observation keywords" in caplog.text
    assert "VALUE" in caplog.text
    assert "DATA" in caplog.text
    assert "ERROR" in caplog.text
    assert "RESTART" in caplog.text


def test_that_general_observations_are_instantiated_with_localization_attributes():
    obs_config_contents = """
        GENERAL_OBSERVATION OBS1 {
            VALUE = 1;
            DATA = GEN;
            ERROR = 0.1;
            RESTART = 1;
        };"""
    ert_config = ert_config_from_parser(obs_config_contents)
    gen_obs = create_observation_dataframes(
        observations=ert_config.observation_declarations, rft_config=None
    )["gen_data"]
    for loc_kw in ["east", "north", "radius"]:
        assert loc_kw in gen_obs.columns
        assert gen_obs[loc_kw].dtype == pl.Float32
        assert gen_obs[loc_kw].to_list() == [None]
