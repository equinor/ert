import os
from contextlib import suppress
from pathlib import Path
from textwrap import dedent

import hypothesis.extra.lark as stlark
import pytest
from hypothesis import given

from ert.config._observations import (
    GeneralObservation,
    HistoryObservation,
    RFTObservation,
    Segment,
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
            ({"type": ObservationType.HISTORY, "name": "FOPR"}),
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
            {
                "type": ObservationType.HISTORY,
                "name": "FWPR",
                "ERROR": "0.1",
                "segments": [("SEG", {"START": "1", "STOP": "0", "ERROR": "0.25"})],
            },
        ],
    ) == [
        HistoryObservation(
            name="FOPR",
            error_mode="RELMIN",
            error=0.1,
            error_min=0.1,
            segments=[],
        ),
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
        HistoryObservation(
            name="FWPR",
            error_mode="RELMIN",
            error=0.1,
            error_min=0.1,
            segments=[
                Segment(
                    name="SEG",
                    start=1,
                    stop=0,
                    error_mode="RELMIN",
                    error=0.25,
                    error_min=0.1,
                ),
            ],
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


@pytest.mark.usefixtures("use_tmpdir")
def test_rft_observation_csv_declaration():
    Path("rft_observations.csv").write_text(
        dedent(
            """
            WELL_NAME,DATE,MD,ZONE,PRESSURE,ERROR,TVD,NORTH,EAST,rms_cell_index,rms_cell_zone_val,rms_cell_zone_str
            WELL1,2013-03-31,2500,zone1,294.0,10,2000.0,71.0,30.0,123,1,zone1
            WELL2,2013-04-30,2600,zone2,295.0,11,2100.0,72.0,31.0,124,2,zone2
            """
        ),
        encoding="utf8",
    )
    assert make_observations(
        "",
        [
            {
                "type": ObservationType.RFT,
                "name": "NAME",
                "CSV": "rft_observations.csv",
            }
        ],
    ) == [
        RFTObservation(
            name="NAME[0]",
            well="WELL1",
            date="2013-03-31",
            value=294.0,
            error=10.0,
            property="PRESSURE",
            north=71.0,
            east=30.0,
            tvd=2000.0,
        ),
        RFTObservation(
            name="NAME[1]",
            well="WELL2",
            date="2013-04-30",
            value=295.0,
            error=11.0,
            property="PRESSURE",
            north=72.0,
            east=31.0,
            tvd=2100.0,
        ),
    ]


@pytest.mark.usefixtures("use_tmpdir")
def test_that_property_can_be_specified_for_rft_observation_csv_declaration():
    Path("rft_observations.csv").write_text(
        dedent(
            """
            WELL_NAME,DATE,MD,ZONE,SWAT,ERROR,TVD,NORTH,EAST,rms_cell_index,rms_cell_zone_val,rms_cell_zone_str
            WELL1,2013-03-31,2500,zone1,0.3,10,2000.0,71.0,30.0,123,1,zone1
            """
        ),
        encoding="utf8",
    )
    assert make_observations(
        "",
        [
            {
                "type": ObservationType.RFT,
                "name": "NAME",
                "CSV": "rft_observations.csv",
                "PROPERTY": "SWAT",
            }
        ],
    ) == [
        RFTObservation(
            name="NAME[0]",
            well="WELL1",
            date="2013-03-31",
            value=0.3,
            error=10.0,
            property="SWAT",
            north=71.0,
            east=30.0,
            tvd=2000.0,
        )
    ]


@pytest.mark.usefixtures("use_tmpdir")
def test_that_missing_columns_in_rft_observations_file_raises_error():
    Path("rft_observations.csv").write_text(
        dedent(
            """
            WELL,DAY,MD,ZONE,PRESSURE,ERR,Z,X,Y,rms_cell_index,rms_cell_zone_val,rms_cell_zone_str
            WELL1,2013-03-31,2500,zone1,294.0,10,2000.0,71.0,30.0,123,1,zone1
            """
        ),
        encoding="utf8",
    )
    with pytest.raises(ObservationConfigError) as err:
        make_observations(
            "",
            [
                {
                    "type": ObservationType.RFT,
                    "name": "NAME",
                    "CSV": "rft_observations.csv",
                }
            ],
        )

    assert (
        "The rft observations file rft_observations.csv is missing required columns"
        " DATE, EAST, ERROR, NORTH, TVD, WELL_NAME." in str(err.value)
    )


def test_that_multiple_segments_are_collected():
    observations = make_and_parse_observations(
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

    assert len(observations[0].segments) == 2
