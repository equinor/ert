import pytest

from ert.parsing.new_observations_parser import (
    ObservationConfigError,
    ObservationType,
    _parse_content,
)


def test_parse():
    assert (
        _parse_content(
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
           DATE       = 2015-06-13;  -- (RESTART = 199)
           OBS_FILE   = wpr_diff_obs.txt;
        };

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
            "",
        )
        == [
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
                ObservationType.HISTORY,
                "FOPR",
                {
                    "ERROR": "0.1",
                    "SEGMENT": ("SEG", {"START": "1", "STOP": "0", "ERROR": "-1"}),
                },
            ),
        ]
    )


def test_parse_include_outside_raises_obsconf_error():
    with pytest.raises(
        ObservationConfigError,
        match="Observation parsing failed: Did not expect character",
    ):
        _parse_content(
            """
            include a;
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
           DATE       = 2015-06-13;  -- (RESTART = 199)
           OBS_FILE   = wpr_diff_obs.txt;
        };

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
            "",
        )
