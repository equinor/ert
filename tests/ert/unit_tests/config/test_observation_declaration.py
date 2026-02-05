import os
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from textwrap import dedent

import hypothesis.extra.lark as stlark
import pytest
from hypothesis import given
from resdata.summary import Summary

from ert.__main__ import run_convert_observations
from ert.config._observations import (
    BreakthroughObservation,
    GeneralObservation,
    RFTObservation,
    SummaryObservation,
    make_observations,
)
from ert.config.observation_config_migrations import HistoryObservation
from ert.config.parsing import parse_observations
from ert.config.parsing.observations_parser import (
    ObservationConfigError,
    ObservationType,
    observations_parser,
)
from ert.namespace import Namespace

observation_contents = stlark.from_lark(observations_parser)


def make_and_parse_observations(contents, filename):
    return make_observations(
        os.path.dirname(filename), parse_observations(contents, filename)
    )


@pytest.mark.slow
@given(observation_contents)
def test_parsing_contents_succeeds_or_gives_config_error(contents):
    with suppress(ObservationConfigError):
        _ = make_and_parse_observations(contents, "observations.txt")


@pytest.mark.usefixtures("use_tmpdir")
def test_make_observations():
    Path("wpr_diff_idx.txt").write_text("400\n800\n1200\n1800\n", encoding="utf8")
    Path("wpr_diff_obs.txt").write_text(
        "1.1 0.1\n2.2 0.2\n3.3 0.3\n4.4 0.4\n", encoding="utf8"
    )

    obs_config_contents = dedent(
        """
        HISTORY_OBSERVATION FOPR {};

        SUMMARY_OBSERVATION WOPR_OP1_9 {
            VALUE = 0.1;
            ERROR = 0.05;
            DATE = 2010-03-31;
            KEY = WOPR:OP1;
        };

        GENERAL_OBSERVATION WPR_DIFF_1 {
            DATA = SNAKE_OIL_WPR_DIFF;
            INDEX_LIST = 400,800,1200,1800;
            DATE = 2015-06-13;
            OBS_FILE = wpr_diff_obs.txt;
        };

        GENERAL_OBSERVATION WPR_DIFF_2 {
            DATA = SNAKE_OIL_WPR_DIFF;
            INDEX_FILE = wpr_diff_idx.txt;
            DATE = 2015-06-13;
            OBS_FILE = wpr_diff_obs.txt;
        };

        HISTORY_OBSERVATION FWPR {
            ERROR = 0.1;
            SEGMENT SEG {
                START = 1;
                STOP = 0;
                ERROR = 0.25;
            };
        };
        """
    )

    Path("obs_config").write_text(obs_config_contents, encoding="utf8")

    # Create a simple refcase so the migration can read history values
    summary = Summary.writer("MY_REFCASE", datetime(2000, 1, 1), 10, 10, 10)
    summary.add_variable("FOPR", unit="SM3/DAY")
    summary.add_variable("FOPRH", unit="SM3/DAY")
    summary.add_variable("FWPR", unit="SM3/DAY")
    summary.add_variable("FWPRH", unit="SM3/DAY")

    # Create two timesteps: the explicit dates used in the test
    start_date = datetime(2010, 3, 31)
    # overwrite writer start date by recreating with desired start
    summary = Summary.writer("MY_REFCASE", start_date, 10, 10, 10)
    summary.add_variable("FOPR", unit="SM3/DAY")
    summary.add_variable("FOPRH", unit="SM3/DAY")
    summary.add_variable("FWPR", unit="SM3/DAY")
    summary.add_variable("FWPRH", unit="SM3/DAY")

    # first step: start_date (2010-03-31)
    t0 = summary.addTStep(1, sim_days=0)
    t0["FOPR"] = 1
    t0["FOPRH"] = 2
    t0["FWPR"] = 3
    t0["FWPRH"] = 4

    # second step: 2015-06-13
    second_date = datetime(2015, 6, 13)
    days_between = (second_date - start_date).days
    t1 = summary.addTStep(1, sim_days=days_between)
    t1["FOPR"] = 1
    t1["FOPRH"] = 2
    t1["FWPR"] = 3
    t1["FWPRH"] = 4

    summary.fwrite()

    config_content = dedent(
        """
        NUM_REALIZATIONS 1
        ECLBASE BASEBASEBASE
        REFCASE MY_REFCASE
        SUMMARY *
        GEN_DATA GEN RESULT_FILE:gen%%d.txt REPORT_STEPS:1
        OBS_CONFIG obs_config
        """
    )
    Path("config.ert").write_text(config_content, encoding="utf8")

    run_convert_observations(Namespace(config="config.ert"))

    # Re-parse the migrated obs_config and build the observation objects
    migrated_contents = Path("obs_config").read_text(encoding="utf8")
    parsed = parse_observations(migrated_contents, "obs_config")
    observations = make_observations(os.path.dirname("obs_config"), parsed)

    # Validate migrated observations contain expected entries and values
    names = [getattr(o, "name", None) for o in observations]
    assert "WOPR_OP1_9" in names
    assert "WPR_DIFF_1" in names
    assert "WPR_DIFF_2" in names
    assert "FOPR" in names
    assert "FWPR" in names

    # Check specific properties
    wopr = next(o for o in observations if getattr(o, "name", None) == "WOPR_OP1_9")
    assert isinstance(wopr, SummaryObservation)
    assert wopr.value == 0.1
    assert wopr.key == "WOPR:OP1"
    assert wopr.date == "2010-03-31"

    wpr1 = next(o for o in observations if getattr(o, "name", None) == "WPR_DIFF_1")
    assert isinstance(wpr1, GeneralObservation)
    # Migration converts DATE -> RESTART, so accept either a date string or a restart
    assert wpr1.restart == 1
    assert wpr1.data == "SNAKE_OIL_WPR_DIFF"

    wpr2 = next(o for o in observations if getattr(o, "name", None) == "WPR_DIFF_2")
    assert isinstance(wpr2, GeneralObservation)
    assert wpr2.restart == 1
    assert wpr2.data == "SNAKE_OIL_WPR_DIFF"


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
            zone="zone1",
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
            zone="zone2",
        ),
    ]


@pytest.mark.usefixtures("use_tmpdir")
def test_that_rft_observations_from_csv_with_no_rows_after_header_returns_empty_list():
    Path("rft_observations.csv").write_text(
        "WELL_NAME,DATE,MD,ZONE,PRESSURE,ERROR,TVD,NORTH,EAST,rms_cell_index,rms_cell_zone_val,rms_cell_zone_str",
        encoding="utf8",
    )
    assert (
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
        == []
    )


@pytest.mark.usefixtures("use_tmpdir")
def test_that_observation_type_rft_is_compatible_with_create_rft_ertobs_handling_of_missing_data():  # noqa: E501
    """A value of -1 and error of 0 is used by fmu.tools.rms create_rft_ertobs to
    indicate missing data. If encountered in an rft observations csv file
    it should be skipped and create a user warning.
    """
    Path("rft_observations.csv").write_text(
        dedent(
            """
            WELL_NAME,DATE,MD,ZONE,PRESSURE,ERROR,TVD,NORTH,EAST,rms_cell_index,rms_cell_zone_val,rms_cell_zone_str
            WELL1,2013-03-31,2500,zone1,-1,0,2000.0,71.0,30.0,123,1,zone1
            WELL1,2013-04-30,2500,zone1,295,10,2000.0,71.0,30.0,123,1,zone1
            WELL2,2014-03-31,2500,zone1,-1,0,2000.0,71.0,30.0,123,1,zone1
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
    assert dedent(
        """
            Invalid value=-1 and error=0 detected in rft_observations.csv for well(s):
             - WELL1 at date 2013-03-31
             - WELL2 at date 2014-03-31
            The invalid observation(s) must be removed from the file.
            """
    ).strip() in str(err.value)


@pytest.mark.usefixtures("use_tmpdir")
def test_that_invalid_numeric_values_in_rft_observations_csv_raises_error():
    Path("rft_observations.csv").write_text(
        dedent(
            """
            WELL_NAME,DATE,MD,ZONE,PRESSURE,ERROR,TVD,NORTH,EAST,rms_cell_index,rms_cell_zone_val,rms_cell_zone_str
            WELL1,2013-03-31,2500,zone1,invalid_value,invalid_value,2000.0,71.0,30.0,123,1,zone1
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
        'Could not convert invalid_value to float. Failed to validate "invalid_value"'
        in str(err.value)
    )


@pytest.mark.usefixtures("use_tmpdir")
def test_that_non_existent_rft_observations_csv_file_raises_error():
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
        "The CSV file (rft_observations.csv) does not exist or is not accessible."
        in str(err.value)
    )


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
            zone="zone1",
        )
    ]


@pytest.mark.usefixtures("use_tmpdir")
def test_that_missing_user_specified_property_raises_error():
    Path("rft_observations.csv").write_text(
        dedent(
            """
            WELL_NAME,DATE,MD,ZONE,PRESSURE,ERROR,TVD,NORTH,EAST,rms_cell_index,rms_cell_zone_val,rms_cell_zone_str
            WELL1,2013-03-31,2500,zone1,0.3,10,2000.0,71.0,30.0,123,1,zone1
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
                    "PROPERTY": "SWAT",
                }
            ],
        )

    assert (
        "rft observations file rft_observations.csv is missing required column(s) SWAT"
        in str(err.value)
    )


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
        "The rft observations file rft_observations.csv is missing required column(s)"
        " DATE, EAST, ERROR, NORTH, TVD, WELL_NAME." in str(err.value)
    )


@pytest.mark.usefixtures("use_tmpdir")
def test_that_multiple_segments_are_collected():
    obs_config_str = """
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
    """

    Path("obs_config.txt").write_text(obs_config_str, encoding="utf8")
    parsed_obs_dict = parse_observations(obs_config_str, "obs_config.txt")
    observations = HistoryObservation.from_obs_dict("", parsed_obs_dict[0])

    assert len(observations[0].segments) == 2


@pytest.mark.usefixtures("use_tmpdir")
def test_that_breakthrough_observation_can_be_instantiated_from_config():
    obs_config_str = """
      BREAKTHROUGH_OBSERVATION name {
        KEY=WWCT:OP_1;
        DATE=2012-10-01;
        ERROR=3; -- days
        THRESHOLD=0.1;
      };
    """

    Path("obs_config.txt").write_text(obs_config_str, encoding="utf8")
    parsed_obs_dict = parse_observations(obs_config_str, "obs_config.txt")
    brt_obs = BreakthroughObservation.from_obs_dict("", parsed_obs_dict[0]).pop()
    assert brt_obs.type == "breakthrough"
    assert brt_obs.name == "name"
    assert brt_obs.response_key == "WWCT:OP_1"
    assert brt_obs.date == datetime.fromisoformat("2012-10-01")
    assert brt_obs.error == 3
    assert brt_obs.threshold == 0.1
    assert brt_obs.east is None
    assert brt_obs.north is None
    assert brt_obs.radius is None


@pytest.mark.parametrize("missing_keyword", ["KEY", "DATE", "ERROR", "THRESHOLD"])
@pytest.mark.usefixtures("use_tmpdir")
def test_that_breakthrough_observation_raises_error_when_missing_required_keyword(
    missing_keyword,
):
    obs_config_str = """
      BREAKTHROUGH_OBSERVATION BRT_OBS {
        KEY=WWCT:OP_1;
        DATE=2012-10-01;
        ERROR=3; -- days
        THRESHOLD=0.1;
      };
    """
    obs_config_lines = obs_config_str.splitlines()
    obs_config_lines.pop(
        next(i for i, line in enumerate(obs_config_lines) if missing_keyword in line)
    )
    obs_config_str = "\n".join(obs_config_lines)
    Path("obs_config.txt").write_text(obs_config_str, encoding="utf8")
    parsed_obs_dict = parse_observations(obs_config_str, "obs_config.txt")
    with pytest.raises(
        ObservationConfigError, match=f'Missing item "{missing_keyword}" in BRT_OBS'
    ):
        BreakthroughObservation.from_obs_dict("", parsed_obs_dict[0])


@pytest.mark.usefixtures("use_tmpdir")
def test_that_breakthrough_observation_can_be_instantiated_with_localization():
    obs_config_str = """
      BREAKTHROUGH_OBSERVATION name {
        KEY=WWCT:OP_1;
        DATE=2012-10-01;
        ERROR=3; -- days
        THRESHOLD=0.1;
        LOCALIZATION {
           EAST=10;
           NORTH=20;
           RADIUS=2500;
        };
      };
    """

    Path("obs_config.txt").write_text(obs_config_str, encoding="utf8")
    parsed_obs_dict = parse_observations(obs_config_str, "obs_config.txt")
    brt_obs = BreakthroughObservation.from_obs_dict("", parsed_obs_dict[0]).pop()
    assert brt_obs.east == 10
    assert brt_obs.north == 20
    assert brt_obs.radius == 2500
