import os
from contextlib import suppress
from pathlib import Path

import hypothesis.extra.lark as stlark
import pytest
from hypothesis import given

from ert.config._observation_declaration import (
    GenObsValues,
    HistoryValues,
    Segment,
    SummaryValues,
    make_observation_declarations,
)
from ert.config.parsing import parse_observations
from ert.config.parsing.observations_parser import (
    ObservationConfigError,
    ObservationType,
    observations_parser,
)

observation_contents = stlark.from_lark(observations_parser)


def parse_observation_declarations(contents, filename):
    return make_observation_declarations(
        os.path.dirname(filename), parse_observations(contents, filename)
    )


@pytest.mark.integration_test
@given(observation_contents)
def test_parsing_contents_succeeds_or_gives_config_error(contents):
    with suppress(ObservationConfigError):
        _ = parse_observation_declarations(contents, "observations.txt")


@pytest.mark.usefixtures("use_tmpdir")
def test_make_observation_declarations():
    Path("wpr_diff_idx.txt").write_text("", encoding="utf8")
    Path("wpr_diff_obs.txt").write_text("", encoding="utf8")
    assert make_observation_declarations(
        "",
        [
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
        ],
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


def test_that_multiple_segments_are_collected():
    observations = parse_observation_declarations(
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
