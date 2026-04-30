from pathlib import Path
from textwrap import dedent

import pytest

from ert.config import ConfigValidationError, ErtConfig
from ert.config._create_observation_dataframes import create_observation_dataframes
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


def test_that_summary_observations_can_be_initialized_from_common_configuration(tmpdir):
    common_obs_config = """
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
    with tmpdir.as_cwd():
        Path("summary_values.csv").write_text(csv_content(), encoding="utf-8")

        common_observations = create_observations(config(), common_obs_config)
        regular_observations = create_observations(config(), regular_obs_config)

        assert regular_observations.keys() == common_observations.keys()
        for obs_type, obs_df in regular_observations.items():
            assert obs_df.equals(common_observations[obs_type])


def test_that_misconfigured_breakthrough_in_summary_gets_breakthrough_label_in_error(
    tmpdir,
):
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

    with tmpdir.as_cwd():
        Path("summary_values.csv").write_text(csv_content(), encoding="utf-8")
        with pytest.raises(
            ConfigValidationError,
            match=r'Line 2 \(Column 5-12\): Missing item "DATE" in BREAKTHROUGH',
        ):
            create_observations(config(), misconfigured_breakthrough_config)


def test_that_common_summary_config_can_be_initialized_without_localization(
    tmpdir,
):
    misconfigured_breakthrough_config = """
    SUMMARY {
        VALUES = summary_values.csv;
        WELL OP1 {};
    };"""

    with tmpdir.as_cwd():
        Path("summary_values.csv").write_text(csv_content(), encoding="utf-8")
        obs = create_observations(config(), misconfigured_breakthrough_config)
        for loc_kw in ["east", "north", "radius"]:
            assert list(obs["summary"][loc_kw]) == [None, None, None]


def test_that_missing_values_in_common_summary_config_raises_config_error(
    tmpdir,
):
    config_without_values = """
    SUMMARY {
        WELL OP1 {};
    };"""

    with tmpdir.as_cwd():
        Path("summary_values.csv").write_text(csv_content(), encoding="utf-8")
        with pytest.raises(
            ConfigValidationError,
            match=r'Line 2 \(Column 5-12\): Missing item "VALUES" in SUMMARY',
        ):
            create_observations(config(), config_without_values)


def test_that_multiple_common_summary_configs_can_be_in_the_same_config(
    tmpdir,
):
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

    with tmpdir.as_cwd():
        Path("summary_values.csv").write_text(csv_content(), encoding="utf-8")

        obs = create_observations(config(), multiple_summary_config)
        assert len(obs["breakthrough"]) == 2
        assert len(obs["summary"]) == 6


def test_that_negative_error_value_in_common_summary_config_raises_config_error(
    tmpdir,
):
    multiple_summary_config = """
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
    csv_content = dedent(
        """well, keyword, value, error, date
        OP1, WOPR, 1e6, -1.0, 2012-02-01
        ,FOPR, 3e6, 3.0, 2012-03-01"""
    )
    with tmpdir.as_cwd():
        Path("summary_values.csv").write_text(csv_content, encoding="utf-8")

        with pytest.raises(
            ConfigValidationError,
            match=r'Failed to validate "-1.0" in \(WOPR:OP1\) ERROR=-1.0. '
            r"\(WOPR:OP1\) ERROR must be given a strictly positive value",
        ):
            create_observations(config(), multiple_summary_config)


def test_that_breakthrough_can_be_instantiated_without_well_localization_through_summary_common_config(  # noqa: E501
    tmpdir,
):
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
    with tmpdir.as_cwd():
        Path("summary_values.csv").write_text(csv_content, encoding="utf-8")
        create_observations(config(), obs_config)


def test_that_summary_common_config_resolves_csv_in_subdirectory(tmpdir):
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
    with tmpdir.as_cwd():
        Path("subdir").mkdir()
        Path("subdir/summary_values.csv").write_text(csv_content(), encoding="utf-8")

        obs = create_observations(config(), obs_config)
        assert len(obs["summary"]) == 3
        assert len(obs["breakthrough"]) == 1


def test_that_summary_common_config_resolves_csv_in_subdirectory_from_main_config(
    tmpdir,
):
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
    with tmpdir.as_cwd():
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


def test_that_breakthrough_in_common_summary_config_doesnt_append_well_name_if_well_name_is_in_breakthrough_key(  # noqa: E501
    tmpdir,
):
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

    with tmpdir.as_cwd():
        Path("summary_values.csv").write_text(csv_content(), encoding="utf-8")

        obs = create_observations(config(), obs_config)
        assert obs["breakthrough"]["response_key"].to_list() == [
            "BREAKTHROUGH:WWCT:OP1"
        ]


def test_that_summary_common_config_validates_strictly_positive_errors(
    tmpdir,
):
    obs_config = """
    SUMMARY {
        VALUES = summary_values.csv;
        WELL OP1 {};
    };
    """
    csv_content = dedent(
        """well, keyword, value, error, date
        OP1, WOPR, 1e6, 0, 2012-02-01"""
    )
    with tmpdir.as_cwd():
        Path("summary_values.csv").write_text(csv_content, encoding="utf-8")
        with pytest.raises(
            ConfigValidationError,
            match=(
                r'Failed to validate "0" in \(WOPR:OP1\) ERROR=0. '
                r"\(WOPR:OP1\) ERROR must be given a strictly positive value."
            ),
        ):
            create_observations(config(), obs_config)


@pytest.mark.parametrize("well_name", ["OP-1", "OP:1", "OP.1", "OP_1-A.B:C"])
def test_that_summary_common_config_handles_special_characters_in_well_name(
    tmpdir, well_name
):
    obs_config = (
        """
    SUMMARY {
        VALUES = summary_values.csv;
        """
        f"WELL {well_name + '{};'}"
        "};"
    )

    special_csv = dedent(
        f"""well, keyword, value, error, date
        {well_name}, WOPR, 1e6, 1.0, 2012-02-01
        ,FOPR, 3e6, 3.0, 2012-03-01"""
    )
    with tmpdir.as_cwd():
        Path("summary_values.csv").write_text(special_csv, encoding="utf-8")

        obs = create_observations(config(), obs_config)
        assert len(obs["summary"]) == 2
        assert obs["summary"]["observation_key"].to_list() == [
            f"WOPR:{well_name}",
            "FOPR",
        ]


def test_that_summary_common_config_raises_error_regarding_required_csv_columns(
    tmpdir,
):
    obs_config = """
    SUMMARY {
        VALUES = summary_values.csv;
        WELL OP1 {};
    };
    """
    csv_content = dedent(
        """keyword, value, error, date
        FOPR, 10, 1, 2012-02-01"""
    )
    with tmpdir.as_cwd():
        Path("summary_values.csv").write_text(csv_content, encoding="utf-8")
        with pytest.raises(
            ConfigValidationError,
            match=r"Line 2 \(Column 5-12\): Missing column "
            '"well" in csv file "summary_values.csv"',
        ):
            create_observations(config(), obs_config)


def test_that_optional_csv_columns_are_included_in_summary_observation_name(tmpdir):
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
    with tmpdir.as_cwd():
        Path("summary_values.csv").write_text(csv_content, encoding="utf-8")
        obs = create_observations(config(), obs_config)
        names = obs["summary"]["observation_key"].to_list()
        assert names == ["BPR:4,4,13", "COFR:OP1:2,5,1"]
