from contextlib import suppress
from pathlib import Path

import hypothesis.extra.lark as stlark
import pytest
from hypothesis import given

from ert.config.parsing import parse_observations
from ert.config.parsing.observations_parser import (
    GenObsValues,
    HistoryValues,
    ObservationConfigError,
    ObservationType,
    Segment,
    SummaryValues,
    _parse_content_list,
    _validate_conf_content,
    observations_parser,
)

observation_contents = stlark.from_lark(observations_parser)


@pytest.mark.integration_test
@given(observation_contents)
def test_parsing_contents_succeeds_or_gives_config_error(contents):
    with suppress(ObservationConfigError):
        _ = _validate_conf_content(
            ".", _parse_content_list(contents, "observations.txt")
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
    assert _parse_content_list(
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
        _parse_content_list(content="include a;", filename="")


def test_that_double_comments_are_handled():
    assert _parse_content_list(
        """
            SUMMARY_OBSERVATION -- foo -- bar -- baz
                        FOPR;
            """,
        "",
    ) == [(ObservationType.SUMMARY, "FOPR")]


@pytest.mark.usefixtures("use_tmpdir")
def test_validate(file_contents):
    Path("wpr_diff_idx.txt").write_text("", encoding="utf8")
    Path("wpr_diff_obs.txt").write_text("", encoding="utf8")
    assert _validate_conf_content(
        "",
        _parse_content_list(
            file_contents,
            "",
        ),
    ) == [
        (
            "FOPR",
            HistoryValues(
                key="FOPR", error_mode="RELMIN", error=0.1, error_min=0.1, segment=[]
            ),
        ),
        (
            "WOPR_OP1_9",
            SummaryValues(
                error_mode="ABS",
                error=0.05,
                error_min=0.1,
                key="WOPR:OP1",
                value=0.1,
                date="2010-03-31",
            ),
        ),
        (
            "WPR_DIFF_1",
            GenObsValues(
                data="SNAKE_OIL_WPR_DIFF",
                index_list="400,800,1200,1800",
                date="2015-06-13",
                obs_file="wpr_diff_obs.txt",
            ),
        ),
        (
            "WPR_DIFF_2",
            GenObsValues(
                data="SNAKE_OIL_WPR_DIFF",
                index_file="wpr_diff_idx.txt",
                date="2015-06-13",
                obs_file="wpr_diff_obs.txt",
            ),
        ),
        (
            "FWPR",
            HistoryValues(
                key="FWPR",
                error_mode="RELMIN",
                error=0.1,
                error_min=0.1,
                segment=[
                    (
                        "SEG",
                        Segment(
                            start=1,
                            stop=0,
                            error_mode="RELMIN",
                            error=0.25,
                            error_min=0.1,
                        ),
                    )
                ],
            ),
        ),
    ]


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
    obs_type, obs_content, match
):
    additional = (
        ""
        if obs_type == "HISTORY_OBSERVATION"
        else "RESTART = 1; VALUE=1.0; KEY = FOPR;"
    )
    with pytest.raises(ObservationConfigError, match=match):
        parse_observations(
            f"""
                        {obs_type}  FOPR
                        {{
                            {obs_content}
                            {additional}
                        }};
                        """,
            "",
        )


@pytest.mark.parametrize(
    "obs_content, match",
    [
        (
            """
            SUMMARY_OBSERVATION  FOPR
            {
               DAYS       = -1;
            };
            """,
            'Failed to validate "-1"',
        ),
        (
            """
            SUMMARY_OBSERVATION  FOPR
            {
               VALUE       = exactly_1;
            };
            """,
            'Failed to validate "exactly_1"',
        ),
        (
            """
            SUMMARY_OBSERVATION  FOPR
            {
               DAYS       = 1;
            };
            """,
            'Missing item "VALUE"',
        ),
        (
            """
            SUMMARY_OBSERVATION  FOPR
            {
               KEY        = FOPR;
               VALUE      = 2.0;
               DAYS       = 1;
            };
            """,
            'Missing item "ERROR"',
        ),
        (
            """
            SUMMARY_OBSERVATION  FOPR
            {
               VALUE = 1;
               ERROR = 0.1;
            };
            """,
            'Missing item "KEY"',
        ),
        (
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
            """,
            'Missing item "START"',
        ),
        (
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
            """,
            'Missing item "STOP"',
        ),
        (
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
            """,
            'Failed to validate "3.2"',
        ),
        (
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
            """,
            'Failed to validate "1.1"',
        ),
        (
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
            """,
            'Failed to validate "-1"',
        ),
        (
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
            """,
            'Failed to validate "-1"',
        ),
        (
            """
            SUMMARY_OBSERVATION  FOPR
            {
               RESTART = -1;
            };
            """,
            'Failed to validate "-1"',
        ),
        (
            """
            SUMMARY_OBSERVATION  FOPR
            {
               RESTART = minus_one;
            };
            """,
            'Failed to validate "minus_one"',
        ),
        (
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
                """,
            'Failed to validate "NOT_ABS"',
        ),
    ],
)
def test_that_summary_observation_validation_is_handled(obs_content, match):
    with pytest.raises(ObservationConfigError, match=match):
        parse_observations(obs_content, filename="")


def test_that_general_observation_without_error_is_invalid():
    with pytest.raises(ObservationConfigError, match="ERROR must also be given"):
        parse_observations(
            """
            GENERAL_OBSERVATION  obs
            {
               DATA       = RES;
               DATE       = 2023-02-01;
               VALUE      = 1;
            };
            """,
            "",
        )


def test_that_general_observation_without_data_is_invalid():
    with pytest.raises(ObservationConfigError, match='Missing item "DATA"'):
        parse_observations(
            """
            GENERAL_OBSERVATION  obs
            {
               DATE       = 2023-02-01;
               VALUE      = 1;
               ERROR      = 0.01;
               ERROR_MIN  = 0.1;
            };
            """,
            "",
        )


@pytest.mark.parametrize(
    "observation_type",
    ["HISTORY_OBSERVATION", "SUMMARY_OBSERVATION", "GENERAL_OBSERVATION"],
)
def test_that_unknown_key_is_handled(observation_type):
    with pytest.raises(ObservationConfigError, match="Unknown SMERROR"):
        parse_observations(f"{observation_type} FOPR {{SMERROR=0.1;DATA=key;}};", "")


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


def test_that_multiple_segments_are_collected():
    observations = parse_observations(
        """
  HISTORY_OBSERVATION GWIR:FIELD
  {
     ERROR       = 0.20;
     ERROR_MODE  = RELMIN;
     ERROR_MIN   = 100;

     SEGMENT FIRST_YEAR
     {
        START = 0;
        STOP  = 10;
        ERROR = 0.50;
        ERROR_MODE = REL;
     };

     SEGMENT SECOND_YEAR
     {
        START      = 11;
        STOP       = 20;
        ERROR      = 1000;
        ERROR_MODE = ABS;
     };
  };
            """,
        "",
    )

    assert len(observations[0][1].segment) == 2
