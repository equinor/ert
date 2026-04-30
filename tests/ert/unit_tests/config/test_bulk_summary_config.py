import math
from datetime import datetime
from pathlib import Path
from textwrap import dedent

import hypothesis.strategies as st
import polars as pl
import pytest
from hypothesis import given, settings
from polars.testing import assert_frame_equal

from ert.config import ConfigValidationError, ErtConfig
from ert.config._create_observation_dataframes import create_observation_dataframes
from ert.config._observations import strip_dataframe_whitespaces
from tests.ert.unit_tests.config.test_summary_config import create_observations


def config():
    return """
    NUM_REALIZATIONS 3
    ECLBASE ECLIPSE_CASE_%d
    OBS_CONFIG observations
    GEN_KW KW_NAME template.txt kw.txt prior.txt
    RANDOM_SEED 1234
    """


def csv_content():
    return dedent(
        """
        well, keyword, value, error, date
        OP1, WOPR, 1e6, 1.0, 2012-02-01
        ,FOPR, 3e6, 3.0, 2012-03-01
        OP2, WGPR, 1e8, 2.0, 2012-04-01"""
    )


@pytest.mark.usefixtures("use_tmpdir")
def test_that_summary_observations_can_be_initialized_from_bulk_configuration():
    bulk_obs_config = """
    SUMMARY {
        VALUES = summary_values.csv;
        WELL OP1 {
            LOCALIZATION {
                EAST = 4.5;
                NORTH = 12.5;
                RADIUS = 3000;
            };
            BREAKTHROUGH {
                KEY = WWCT;
                THRESHOLD = 0.5;
                DATE = 2020-01-01;
                ERROR = 0.2;
            };
        };
        WELL OP2 {
            LOCALIZATION {
                EAST = 10;
                NORTH = 20;
                RADIUS = 2500;
            };
        };
    };"""

    regular_obs_config = """
    SUMMARY_OBSERVATION {
        KEY = WOPR:OP1;
        VALUE = 1e6;
        ERROR = 1.0;
        DATE = 2012-02-01;
        LOCALIZATION {
            EAST = 4.5;
            NORTH = 12.5;
            RADIUS = 3000;
        };
    };
    SUMMARY_OBSERVATION {
        KEY = FOPR;
        VALUE = 3e6;
        ERROR = 3.0;
        DATE = 2012-03-01;
    };
    SUMMARY_OBSERVATION {
        KEY = WGPR:OP2;
        VALUE = 1e8;
        ERROR = 2.0;
        DATE = 2012-04-01;
        LOCALIZATION {
            EAST = 10.0;
            NORTH = 20.0;
            RADIUS = 2500;
        };
    };
    BREAKTHROUGH_OBSERVATION {
        KEY = WWCT:OP1;
        THRESHOLD = 0.5;
        DATE = 2020-01-01;
        ERROR = 0.2;
            LOCALIZATION {
            EAST = 4.5;
            NORTH = 12.5;
            RADIUS = 3000;
        };
    };
    """
    Path("summary_values.csv").write_text(csv_content(), encoding="utf-8")

    bulk_config_observations = create_observations(config(), bulk_obs_config)
    regular_observations = create_observations(config(), regular_obs_config)

    assert regular_observations.keys() == bulk_config_observations.keys()
    for obs_type, obs_df in regular_observations.items():
        assert obs_df.equals(bulk_config_observations[obs_type])


@pytest.mark.usefixtures("use_tmpdir")
def test_that_misconfigured_breakthrough_in_summary_gets_breakthrough_label_in_error():
    misconfigured_breakthrough_config = """
    SUMMARY {
        VALUES = summary_values.csv;
        WELL OP1 {
            LOCALIZATION {
                EAST = 4.5;
                NORTH = 12.5;
                RADIUS = 3000;
            };
            BREAKTHROUGH {
                THRESHOLD = 0.5;
                KEY = WWCT:OP1;
                -- DATE = 2020-01-01;
                ERROR = 0.2;
            };
        };
    };"""

    Path("summary_values.csv").write_text(csv_content(), encoding="utf-8")
    with pytest.raises(
        ConfigValidationError,
        match=r'Line 2 \(Column 5-12\): Missing item "DATE" in BREAKTHROUGH',
    ):
        create_observations(config(), misconfigured_breakthrough_config)


@pytest.mark.usefixtures("use_tmpdir")
def test_that_bulk_summary_config_can_be_initialized_without_localization():
    misconfigured_breakthrough_config = """
    SUMMARY {
        VALUES = summary_values.csv;
        WELL OP1 {};
    };"""

    Path("summary_values.csv").write_text(csv_content(), encoding="utf-8")
    obs = create_observations(config(), misconfigured_breakthrough_config)
    for loc_kw in ["east", "north", "radius"]:
        assert list(obs["summary"][loc_kw]) == [None, None, None]


@pytest.mark.usefixtures("use_tmpdir")
def test_that_missing_values_in_bulk_summary_config_raises_config_error():
    config_without_values = """
    SUMMARY {
        WELL OP1 {};
    };"""

    Path("summary_values.csv").write_text(csv_content(), encoding="utf-8")
    with pytest.raises(
        ConfigValidationError,
        match=r'Line 2 \(Column 5-12\): Missing item "VALUES" in SUMMARY',
    ):
        create_observations(config(), config_without_values)


@pytest.mark.usefixtures("use_tmpdir")
def test_that_multiple_bulk_summary_configs_can_be_in_the_same_config():
    multiple_summary_config = 2 * (
        """
    SUMMARY {
        VALUES = summary_values.csv;
        WELL OP1 {
            LOCALIZATION {
                EAST = 4.5;
                NORTH = 12.5;
                RADIUS = 3000;
            };
            BREAKTHROUGH {
                THRESHOLD = 0.5;
                DATE = 2020-01-01;
                KEY = WWCT;
                ERROR = 0.2;
            };
        };
    };
    """
    )

    Path("summary_values.csv").write_text(csv_content(), encoding="utf-8")

    obs = create_observations(config(), multiple_summary_config)
    assert len(obs["breakthrough"]) == 2
    assert len(obs["summary"]) == 6


@pytest.mark.usefixtures("use_tmpdir")
def test_that_breakthrough_can_be_instantiated_with_well_localization_through_summary_bulk_config():  # noqa: E501
    obs_config = """
    SUMMARY {
        VALUES = summary_values.csv;
        WELL OP1 {
            BREAKTHROUGH {
                THRESHOLD = 0.5;
                DATE = 2020-01-01;
                KEY = WWCT;
                ERROR = 0.2;
            };
            LOCALIZATION {
                EAST=10;
                NORTH=20;
                RADIUS=30;
            };
        };
    };
    """
    csv_content = dedent(
        """well, keyword, value, error, date
        OP1, WOPR, 1e6, 1.0, 2012-02-01
        ,FOPR, 3e6, 3.0, 2012-03-01"""
    )
    Path("summary_values.csv").write_text(csv_content, encoding="utf-8")
    brt_obs = create_observations(config(), obs_config)["breakthrough"]
    assert brt_obs["east"].to_list() == [10]
    assert brt_obs["north"].to_list() == [20]
    assert brt_obs["radius"].to_list() == [30]


@pytest.mark.usefixtures("use_tmpdir")
def test_that_breakthrough_can_be_instantiated_without_well_localization_through_summary_bulk_config():  # noqa: E501
    obs_config = """
    SUMMARY {
        VALUES = summary_values.csv;
        WELL OP1 {
            BREAKTHROUGH {
                THRESHOLD = 0.5;
                DATE = 2020-01-01;
                KEY = WWCT;
                ERROR = 0.2;
            };
        };
    };
    """
    csv_content = dedent(
        """well, keyword, value, error, date
        OP1, WOPR, 1e6, 1.0, 2012-02-01
        ,FOPR, 3e6, 3.0, 2012-03-01"""
    )
    Path("summary_values.csv").write_text(csv_content, encoding="utf-8")
    create_observations(config(), obs_config)


@pytest.mark.usefixtures("use_tmpdir")
def test_that_summary_bulk_config_resolves_csv_in_subdirectory():
    obs_config = """
    SUMMARY {
        VALUES = subdir/summary_values.csv;
        WELL OP1 {
            LOCALIZATION {
                EAST = 4.5;
                NORTH = 12.5;
                RADIUS = 3000;
            };
            BREAKTHROUGH {
                THRESHOLD = 0.5;
                DATE = 2020-01-01;
                KEY = WWCT;
                ERROR = 0.2;
            };
        };
    };
    """
    Path("subdir").mkdir()
    Path("subdir/summary_values.csv").write_text(csv_content(), encoding="utf-8")

    obs = create_observations(config(), obs_config)
    assert len(obs["summary"]) == 3
    assert len(obs["breakthrough"]) == 1


@pytest.mark.usefixtures("use_tmpdir")
def test_that_summary_bulk_config_resolves_csv_in_subdirectory_from_main_config():
    summary_config = """
    SUMMARY {
        VALUES = summary_values.csv;
        WELL OP1 {
            LOCALIZATION {
                EAST = 4.5;
                NORTH = 12.5;
                RADIUS = 3000;
            };
            BREAKTHROUGH {
                THRESHOLD = 0.5;
                DATE = 2020-01-01;
                KEY = WWCT;
                ERROR = 0.2;
            };
        };
    };
    """
    ert_config = """
    NUM_REALIZATIONS 3
    ECLBASE ECLIPSE_CASE_%d
    OBS_CONFIG subdir/observations
    GEN_KW KW_NAME template.txt kw.txt prior.txt
    RANDOM_SEED 1234
    """
    Path("subdir").mkdir()
    Path("subdir/observations").write_text(summary_config, encoding="utf-8")
    Path("subdir/summary_values.csv").write_text(csv_content(), encoding="utf-8")
    Path("config.ert").write_text(ert_config, encoding="utf-8")
    Path("template.txt").write_text("MY_KEYWORD <MY_KEYWORD>", encoding="utf-8")
    Path("prior.txt").write_text("MY_KEYWORD NORMAL 0 1", encoding="utf-8")

    ert_conf = ErtConfig.from_file("config.ert")
    obs = create_observation_dataframes(
        ert_conf.observation_declarations, None, ert_conf.shape_registry
    )
    assert len(obs["summary"]) == 3
    assert len(obs["breakthrough"]) == 1


@pytest.mark.usefixtures("use_tmpdir")
def test_that_breakthrough_in_bulk_summary_config_doesnt_append_well_name_if_well_name_is_in_breakthrough_key():  # noqa: E501
    obs_config = """
    SUMMARY {
        VALUES = summary_values.csv;
        WELL OP1 {
            LOCALIZATION {
                EAST = 4.5;
                NORTH = 12.5;
                RADIUS = 3000;
            };
            BREAKTHROUGH {
                THRESHOLD = 0.5;
                DATE = 2020-01-01;
                KEY = WWCT:OP1;
                ERROR = 0.2;
            };
        };
    };
    """
    Path("summary_values.csv").write_text(csv_content(), encoding="utf-8")

    obs = create_observations(config(), obs_config)
    assert obs["breakthrough"]["response_key"].to_list() == ["BREAKTHROUGH:WWCT:OP1"]


@pytest.mark.parametrize("error", [-100, -0.5, 0])
@pytest.mark.usefixtures("use_tmpdir")
def test_that_error_value_raises_config_validation_error_if_not_strictly_positive(
    error,
):
    obs_config = """
    SUMMARY {
        VALUES = summary_values.csv;
        WELL OP1 {};
    };
    """
    csv_content = dedent(
        f"""well, keyword, value, error, date
        OP1, WOPR, 1e6, {error}, 2012-02-01"""
    )
    Path("summary_values.csv").write_text(csv_content, encoding="utf-8")
    with pytest.raises(
        ConfigValidationError,
        match=(
            rf'Failed to validate "{error}" in \(WOPR:OP1\) ERROR={error}. '
            r"\(WOPR:OP1\) ERROR must be given a strictly positive value."
        ),
    ):
        create_observations(config(), obs_config)


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize("well_name", ["OP-1", "OP:1", "OP.1", "OP_1-A.B:C"])
def test_that_summary_bulk_config_obs_receives_expected_name_given_special_characters(
    well_name,
):
    obs_config = (
        """
    SUMMARY {
        VALUES = summary_values.csv;
        """
        f"WELL {well_name + '{};'}"
        "};"
    )
    wopr_date = "2012-02-01"
    fopr_date = "2020-01-01"

    special_csv = dedent(
        f"""well, keyword, value, error, date
        {well_name}, WOPR, 1e6, 1.0, {wopr_date}
        ,FOPR, 3e6, 3.0, {fopr_date}"""
    )
    Path("summary_values.csv").write_text(special_csv, encoding="utf-8")

    obs = create_observations(config(), obs_config)
    assert len(obs["summary"]) == 2
    assert obs["summary"]["observation_key"].to_list() == [
        f"WOPR:{well_name}:{wopr_date}",
        f"FOPR:{fopr_date}",
    ]


@pytest.mark.usefixtures("use_tmpdir")
def test_that_summary_bulk_config_raises_error_regarding_required_csv_columns():
    obs_config = """
    SUMMARY {
        VALUES = summary_values.csv;
        WELL OP1 {};
    };
    """
    csv_content = dedent(
        """keyword, error, date
        FOPR, 1, 2012-02-01"""
    )
    Path("summary_values.csv").write_text(csv_content, encoding="utf-8")
    with pytest.raises(
        ConfigValidationError,
        match=r"Line 2 \(Column 5-12\): Missing required column "
        "'value' in csv file 'summary_values.csv'",
    ):
        create_observations(config(), obs_config)


@pytest.mark.usefixtures("use_tmpdir")
def test_that_optional_csv_columns_are_included_in_summary_observation_name():
    obs_config = """
    SUMMARY {
        VALUES = summary_values.csv;
        WELL OP1 {};
    };
    """
    csv_content = dedent(
        """\
        well, keyword, value, error, date, number, nx, ny
        OP1, BPR, 1e6, 1.0, 2012-02-01, 1234, 10, 10
        OP1, COFR, 2e6, 2.0, 2012-03-01, 42, 10, 10"""
    )
    Path("summary_values.csv").write_text(csv_content, encoding="utf-8")
    obs = create_observations(config(), obs_config)
    names = obs["summary"]["observation_key"].to_list()
    assert names == ["BPR:4,4,13:2012-02-01", "COFR:OP1:2,5,1:2012-03-01"]


@given(
    keyword=st.text(
        min_size=1,
        max_size=8,
        alphabet=st.characters(categories=("L", "N", "P"), exclude_characters=","),
    )
)
@pytest.mark.usefixtures("use_tmpdir")
@settings(max_examples=10)
def test_that_fully_populated_csv_does_not_crash_given_arbitrary_keyword(keyword):
    """We expect that when all columns are populated, there should not exist a keyword
    which would crash given too few columns."""
    obs_config = """
    SUMMARY {
        VALUES = summary_values.csv;
        WELL OP1 {};
    };
    """
    csv_columns = (
        "well, keyword, value, error, date, number, nx, ny, lgr_name, li, lj, lk"
    )
    csv_row = f"OP1, {keyword}, 1e6, 1.0, 2012-02-01, 10, 1, 1, foo, 1, 1, 1"
    csv_content_ = f"{csv_columns}\n{csv_row}"

    Path("summary_values.csv").write_text(csv_content_, encoding="utf-8")
    create_observations(config(), obs_config)


@pytest.mark.usefixtures("use_tmpdir")
def test_that_ill_configured_keyword_raises_config_validation_error():
    obs_config = """
    SUMMARY {
        VALUES = summary_values.csv;
        WELL OP1 {};
    };
    """
    csv_content_ = """well, keyword, value, error, date
        "OP1, BPR, 1e6, 1.0, 2012-02-01"""

    Path("summary_values.csv").write_text(csv_content_, encoding="utf-8")
    with pytest.raises(
        ConfigValidationError,
        match=r"Could not create summary key for keyword 'BPR' in file "
        r"'summary_values.csv' \(line 1\)",
    ):
        create_observations(config(), obs_config)


@pytest.mark.usefixtures("use_tmpdir")
def test_that_duplicate_wells_raises_config_validation_error():
    obs_config = """
    SUMMARY {
        VALUES = summary_values.csv;
        WELL OP1 {};
        WELL OP1 {};
    };
    """

    with pytest.raises(
        ConfigValidationError,
        match=r"Observation contains duplicate key WELL OP1",
    ):
        create_observations(config(), obs_config)


def test_that_strip_dataframe_whitespaces_strips_column_names():
    df = pl.DataFrame({" col1 ": ["a"], "  col2": ["b"], "col3  ": ["c"]})
    result = strip_dataframe_whitespaces(df)
    assert result.columns == ["col1", "col2", "col3"]


def test_that_strip_dataframe_whitespaces_strips_string_columns():
    df = pl.DataFrame({"col1": [" hello "], "col2": ["  world  "]})
    result = strip_dataframe_whitespaces(df)
    assert result["col1"].to_list() == ["hello"]
    assert result["col2"].to_list() == ["world"]


def test_that_strip_dataframe_whitespaces_preserves_non_string_columns():
    df = pl.DataFrame({"name": [" foo "], "value": [42], "rate": [math.pi]})
    result = strip_dataframe_whitespaces(df)
    assert result["name"].to_list() == ["foo"]
    assert result["value"].to_list() == [42]
    assert result["rate"].to_list() == [math.pi]


def test_that_strip_dataframe_whitespaces_converts_empty_strings_to_none():
    df = pl.DataFrame({"col1": ["  ", "", " x "], "col2": ["a", "   ", "b"]})
    result = strip_dataframe_whitespaces(df)
    assert result["col1"].to_list() == [None, None, "x"]
    assert result["col2"].to_list() == ["a", None, "b"]


@pytest.mark.usefixtures("use_tmpdir")
def test_that_docs_setup_results_in_expected_observation_dataframes():
    """This test replicates the example made in observations.rst, ensuring
    that our documentation is a working happy path."""
    obs_config = """
    SUMMARY {
        VALUES = summary_values.csv;
        WELL OP1 {
            LOCALIZATION {
                EAST   = 32.132;
                NORTH  = 45.139;
                RADIUS = 2500;
            };
            BREAKTHROUGH {
                KEY       = WWCT;
                THRESHOLD = 0.2;
                DATE      = 2012-05-01;
                ERROR     = 3;
            };
        };
        WELL OP2 {
            LOCALIZATION {
                EAST   = 35.734;
                NORTH  = 42.981;
                RADIUS = 3000;
            };
        };
    };"""
    csv_content_ = """ well,keyword, value, error, date, nx, ny, number
     OP1, WOPR, 1e5, 1e3, 2012-01-03, , ,
     OP1, BPR, 1e5, 1e3, 2012-01-07, 1, 1, 3
     OP2, WOPR, 7e5, 1.2e3, 2012-02-05, , ,
     ,FOPR, 1e6, 1e3, 2012-12-02, , ,"""
    Path("summary_values.csv").write_text(csv_content_, encoding="utf-8")
    result = create_observations(config(), obs_config)

    expected_summary = pl.DataFrame(
        {
            "response_key": ["WOPR:OP1", "BPR:1,1,3", "WOPR:OP2", "FOPR"],
            "observation_key": [
                "WOPR:OP1:2012-01-03",
                "BPR:1,1,3:2012-01-07",
                "WOPR:OP2:2012-02-05",
                "FOPR:2012-12-02",
            ],
            "time": [
                datetime.fromisoformat("2012-01-03"),
                datetime.fromisoformat("2012-01-07"),
                datetime.fromisoformat("2012-02-05"),
                datetime.fromisoformat("2012-12-02"),
            ],
            "observations": [1e5, 1e5, 7e5, 1e6],
            "std": [1e3, 1e3, 1.2e3, 1e3],
            "east": [32.132, 32.132, 35.734001, None],
            "north": [45.139, 45.139, 42.980999, None],
            "radius": [2500.0, 2500.0, 3000.0, None],
        },
        schema={
            "response_key": pl.Utf8,
            "observation_key": pl.Utf8,
            "time": pl.Datetime("ms"),
            "observations": pl.Float32,
            "std": pl.Float32,
            "east": pl.Float32,
            "north": pl.Float32,
            "radius": pl.Float32,
        },
    )
    assert_frame_equal(result["summary"], expected_summary)

    expected_breakthrough = pl.DataFrame(
        {
            "observation_key": ["BREAKTHROUGH:WWCT:OP1:0.2"],
            "response_key": ["BREAKTHROUGH:WWCT:OP1"],
            "time": [datetime.fromisoformat("2012-05-01")],
            "observations": [0.0],
            "threshold": [0.2],
            "std": [3.0],
            "east": [32.132],
            "north": [45.139],
            "radius": [2500.0],
        },
        schema={
            "observation_key": pl.Utf8,
            "response_key": pl.Utf8,
            "time": pl.Datetime("ms"),
            "observations": pl.Float32,
            "threshold": pl.Float64,
            "std": pl.Float32,
            "east": pl.Float32,
            "north": pl.Float32,
            "radius": pl.Float32,
        },
    )
    assert_frame_equal(result["breakthrough"], expected_breakthrough)


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize(
    "missing_col",
    ["keyword", "value", "error", "date"],
)
def test_that_missing_value_in_required_column_raises_config_error(missing_col):
    """If a required column has missing values in the csv, the error message
    reports the column name, offending row numbers, and filename."""
    obs_config = """
    SUMMARY {
        VALUES = summary_values.csv;
    };
    """
    complete_row = {
        "keyword": "BAR",
        "value": "1e6",
        "error": "1e3",
        "date": "2012-01-01",
    }
    rows = [
        {**complete_row, missing_col: ""},
        {**complete_row},
        {**complete_row, missing_col: ""},
    ]
    columns = list(complete_row)
    csv_content_ = ",".join(columns) + "\n"
    csv_content_ += "\n".join(",".join(row[c] for c in columns) for row in rows) + "\n"

    Path("summary_values.csv").write_text(csv_content_, encoding="utf-8")
    with pytest.raises(
        ConfigValidationError,
        match=(
            rf"Missing value for column '{missing_col}' "
            rf"in row\(s\) \['0', '2'\] in csv file 'summary_values.csv'"
        ),
    ):
        create_observations(config(), obs_config)


@pytest.mark.usefixtures("use_tmpdir")
def test_that_invalid_csv_file_raises_config_error():
    obs_config = """
    SUMMARY {
        VALUES = summary_values.csv;
        WELL OP1 {};
    };
    """
    Path("summary_values.csv").write_bytes(b"\x80\x81\x82\xff\xfe\x00\x01")
    with pytest.raises(
        ConfigValidationError,
        match=(
            r"Could not read VALUES csv file "
            r"'.*summary_values\.csv': 'utf-8' codec can't decode byte"
        ),
    ):
        create_observations(config(), obs_config)
