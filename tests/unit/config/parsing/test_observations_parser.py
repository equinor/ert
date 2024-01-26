from pathlib import Path

import pytest

from ert.config.parsing.observations_parser import (
    ObservationConfigError,
    ObservationType,
    _parse_content,
    _validate_conf_content,
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
           DATE       = 2015-06-13;  -- (RESTART = 199)
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
    assert _parse_content(
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
                "SEGMENT": ("SEG", {"START": "1", "STOP": "0", "ERROR": "0.25"}),
            },
        ),
    ]


def test_that_unexpected_character_gives_observation_config_error():
    with pytest.raises(
        ObservationConfigError,
        match=".*i.*line 1.*include a;",
    ):
        _parse_content(content="include a;", filename="")


def test_that_double_comments_are_handled():
    assert (
        _parse_content(
            """
            SUMMARY_OBSERVATION -- foo -- bar -- baz
                        FOPR;
            """,
            "",
        )
        == [(ObservationType.SUMMARY, "FOPR")]
    )


@pytest.mark.usefixtures("use_tmpdir")
def test_validate(file_contents):
    Path("wpr_diff_idx.txt").write_text("", encoding="utf8")
    Path("wpr_diff_obs.txt").write_text("", encoding="utf8")
    print(
        _validate_conf_content(
            "",
            _parse_content(
                file_contents,
                "",
            ),
        )
    )
    assert _validate_conf_content(
        "",
        _parse_content(
            file_contents,
            "",
        ),
    ) == [
        (
            ObservationType.HISTORY,
            "FOPR",
            {"ERROR_MODE": "RELMIN", "ERROR": 0.1, "ERROR_MIN": 0.1, "SEGMENT": []},
        ),
        (
            ObservationType.SUMMARY,
            "WOPR_OP1_9",
            {
                "ERROR_MODE": "ABS",
                "ERROR": 0.05,
                "ERROR_MIN": 0.1,
                "KEY": "WOPR:OP1",
                "VALUE": 0.1,
                "DATE": "2010-03-31",
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
                "ERROR_MODE": "RELMIN",
                "ERROR": 0.1,
                "ERROR_MIN": 0.1,
                "SEGMENT": [
                    (
                        "SEG",
                        {
                            "START": 1,
                            "STOP": 0,
                            "ERROR_MODE": "RELMIN",
                            "ERROR": 0.25,
                            "ERROR_MIN": 0.1,
                        },
                    )
                ],
            },
        ),
    ]
