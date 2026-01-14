import os
from contextlib import suppress
from pathlib import Path

import hypothesis.extra.lark as stlark
import pytest
from hypothesis import given

from ert.config._observations import (
    GeneralObservation,
    RFTObservation,
    SummaryObservation,
    make_observations,
)
from ert.config.parsing import parse_observations
from ert.config.parsing.observations_parser import (
    ObservationConfigError,
    ObservationType,
    observations_parser,
)

observation_contents = stlark.from_lark(observations_parser)


def make_and_parse_observations(contents, filename):
    return make_observations(
        os.path.dirname(filename), parse_observations(contents, filename)
    )


@pytest.mark.integration_test
@given(observation_contents)
def test_parsing_contents_succeeds_or_gives_config_error(contents):
    with suppress(ObservationConfigError):
        _ = make_and_parse_observations(contents, "observations.txt")


@pytest.mark.usefixtures("use_tmpdir")
def test_make_observations():
    Path("wpr_diff_idx.txt").write_text("", encoding="utf8")
    Path("wpr_diff_obs.txt").write_text("", encoding="utf8")
    assert make_observations(
        "",
        [
            {
                "type": ObservationType.SUMMARY,
                "name": "WOPR_OP1_9",
                "VALUE": "0.1",
                "ERROR": "0.05",
                "DATE": "2010-03-31",
                "KEY": "WOPR:OP1",
            },
            {
                "type": ObservationType.GENERAL,
                "name": "WPR_DIFF_1",
                "DATA": "SNAKE_OIL_WPR_DIFF",
                "INDEX_LIST": "400,800,1200,1800",
                "DATE": "2015-06-13",
                "OBS_FILE": "wpr_diff_obs.txt",
            },
            {
                "type": ObservationType.GENERAL,
                "name": "WPR_DIFF_2",
                "DATA": "SNAKE_OIL_WPR_DIFF",
                "INDEX_FILE": "wpr_diff_idx.txt",
                "DATE": "2015-06-13",
                "OBS_FILE": "wpr_diff_obs.txt",
            },
        ],
    ) == [
        SummaryObservation(
            name="WOPR_OP1_9",
            error_mode="ABS",
            error=0.05,
            error_min=0.1,
            key="WOPR:OP1",
            value=0.1,
            date="2010-03-31",
            location_x=None,
            location_y=None,
            location_range=None,
        ),
        GeneralObservation(
            name="WPR_DIFF_1",
            data="SNAKE_OIL_WPR_DIFF",
            index_list="400,800,1200,1800",
            date="2015-06-13",
            obs_file="wpr_diff_obs.txt",
        ),
        GeneralObservation(
            name="WPR_DIFF_2",
            data="SNAKE_OIL_WPR_DIFF",
            index_file="wpr_diff_idx.txt",
            date="2015-06-13",
            obs_file="wpr_diff_obs.txt",
        ),
    ]


def test_rft_observation_declaration():
    assert make_observations(
        "",
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
    ) == [
        RFTObservation(
            name="NAME",
            well="well",
            date="2013-03-31",
            value=700.0,
            error=0.1,
            property="PRESSURE",
            north=71.0,
            east=30.0,
            tvd=2000.0,
        )
    ]
