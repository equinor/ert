from textwrap import dedent

import pytest

from ert.config.parsing import parse_observations
from ert.config.parsing.observations_parser import (
    ObservationConfigError,
    ObservationType,
)


def test_parse_observations():
    assert parse_observations(
        """
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
    """,
        "",
    ) == [
        (ObservationType.HISTORY, "FOPR", {}),
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


def test_that_missing_assignment_in_observation_body_shows_informative_error_message():
    expected_match = (
        r"Line 5 \(Column 4-5\): Expected assignment to property 'A'. Got '}' instead."
    )
    with pytest.raises(ObservationConfigError, match=expected_match):
        parse_observations(
            content=dedent("""\
                GENERAL_OBSERVATION POLY_OBS_0 {
                   DATA       = POLY_RES;
                   INDEX_LIST = 0,2,4,6,8;
                   OBS_FILE   = poly_obs_data.txt;
                   A
                };
            """),
            filename="",
        )


def test_that_misspelled_observation_type_shows_informative_error_message():
    expected_match = (
        r"Line 1 \(Column 1-23\): Unknown observation type 'MISSPELLED_OBSERVATION', "
        r"expected either 'GENERAL_OBSERVATION', "
        r"'SUMMARY_OBSERVATION' or 'HISTORY_OBSERVATION'."
    )
    with pytest.raises(ObservationConfigError, match=expected_match):
        parse_observations(
            content=dedent("""\
                MISSPELLED_OBSERVATION POLY_OBS_0 {
                   DATA       = POLY_RES;
                   INDEX_LIST = 0,2,4,6,8;
                   OBS_FILE   = poly_obs_data.txt;
                };
            """),
            filename="",
        )


def test_that_invalid_start_of_observation_body_symbol_show_informative_error_message():
    expected_match = (
        r"Line 1 \(Column 32-33\): Expected either start of observation body \('{'\) "
        r"or end of observation \(';'\), got '\(' instead."
    )
    with pytest.raises(ObservationConfigError, match=expected_match):
        parse_observations(
            content=dedent("""\
                GENERAL_OBSERVATION POLY_OBS_0 (
                   DATA       = POLY_RES;
                   INDEX_LIST = 0,2,4,6,8;
                   OBS_FILE   = poly_obs_data.txt;
                };
            """),
            filename="",
        )


def test_that_repeated_comments_are_ignored():
    assert parse_observations(
        """
            SUMMARY_OBSERVATION -- foo -- bar -- baz
                        FOPR;
            """,
        "",
    ) == [(ObservationType.SUMMARY, "FOPR", {})]


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
