import pytest

from ert.config.parsing import parse_observations
from ert.config.parsing.observations_parser import (
    ObservationConfigError,
    ObservationType,
)


@pytest.fixture
def file_contents():
    return """
        HISTORY_OBSERVATION FOPR;

        SUMMARY_OBSERVATION WOPR_OP1_9
        {
            VALUE   = 0.1;
            ERROR   = 0.05;
            DATE    = 2010-03-31;  -- (RESTART = 9)
            KEY     = WOPR:OP1;
        };

        GENERAL_OBSERVATION WPR_DIFF_1 {
           DATA       = SNAKE_OIL_WPR_DIFF;
           INDEX_LIST = 400,800,1200,1800;
           DATE       = 2015-06-13;-- (RESTART = 199)
           OBS_FILE   = wpr_diff_obs.txt;
        };


        GENERAL_OBSERVATION WPR_DIFF_2 {
           DATA       = SNAKE_OIL_WPR_DIFF;
           INDEX_FILE = wpr_diff_idx.txt;
           DATE       = 2015-06-13;  -- (RESTART = 199)
           OBS_FILE   = wpr_diff_obs.txt;
        };

        HISTORY_OBSERVATION  FWPR
        {
           ERROR      = 0.1;

           SEGMENT SEG
           {
              START = 1;
              STOP  = 0;
              ERROR = 0.25;
           };
        };--comment
    """


def test_parse(file_contents):
    assert parse_observations(
        file_contents,
        "",
    ) == [
        (ObservationType.HISTORY, "FOPR"),
        (
            ObservationType.SUMMARY,
            "WOPR_OP1_9",
            {
                "VALUE": "0.1",
                "ERROR": "0.05",
                "DATE": "2010-03-31",
                "KEY": "WOPR:OP1",
            },
        ),
        (
            ObservationType.GENERAL,
            "WPR_DIFF_1",
            {
                "DATA": "SNAKE_OIL_WPR_DIFF",
                "INDEX_LIST": "400,800,1200,1800",
                "DATE": "2015-06-13",
                "OBS_FILE": "wpr_diff_obs.txt",
            },
        ),
        (
            ObservationType.GENERAL,
            "WPR_DIFF_2",
            {
                "DATA": "SNAKE_OIL_WPR_DIFF",
                "INDEX_FILE": "wpr_diff_idx.txt",
                "DATE": "2015-06-13",
                "OBS_FILE": "wpr_diff_obs.txt",
            },
        ),
        (
            ObservationType.HISTORY,
            "FWPR",
            {
                "ERROR": "0.1",
                ("SEGMENT", "SEG"): {"START": "1", "STOP": "0", "ERROR": "0.25"},
            },
        ),
    ]


def test_that_unexpected_character_gives_observation_config_error():
    with pytest.raises(
        ObservationConfigError,
        match=r"Line 1.*include a;",
    ):
        parse_observations(content="include a;", filename="")


def test_that_double_comments_are_handled():
    assert parse_observations(
        """
            SUMMARY_OBSERVATION -- foo -- bar -- baz
                        FOPR;
            """,
        "",
    ) == [(ObservationType.SUMMARY, "FOPR")]


def test_unexpected_character_handling():
    with pytest.raises(
        ObservationConfigError,
        match=r"Did not expect character: \$ \(on line 4: *ERROR *\$"
        r" 0.20;\). Expected one of {'EQUAL'}",
    ) as err_record:
        parse_observations(
            """
            GENERAL_OBSERVATION GEN_OBS
            {
               ERROR       $ 0.20;
            };
            """,
            "",
        )

    err = err_record.value.errors[0]
    assert err.line == 4
    assert err.end_line == 4
    assert err.column == 28
    assert err.end_column == 29


@pytest.mark.parametrize("observation_type", ["HISTORY", "GENERAL", "SUMMARY"])
def test_that_duplicate_keys_results_in_error_message_with_location(observation_type):
    with pytest.raises(
        ObservationConfigError,
        match=r"Line 7 \(Column 16-25\): Observation contains duplicate key ERROR_MIN",
    ):
        parse_observations(
            f"""
            {observation_type}_OBSERVATION GWIR:FIELD
            {{
               ERROR       = 0.20;
               ERROR_MODE  = RELMIN;
               ERROR_MIN   = 100;
               ERROR_MIN   = 50;
            }};
            """,
            "",
        )
