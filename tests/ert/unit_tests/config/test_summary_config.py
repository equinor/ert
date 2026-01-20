import os
import re
from contextlib import suppress
from pathlib import Path
from textwrap import dedent

import hypothesis.strategies as st
import polars as pl
import pytest
from hypothesis import given, settings
from resfo_utilities.testing import summaries

from ert.config import (
    ConfigValidationError,
    ErtConfig,
    InvalidResponseFile,
    SummaryConfig,
)
from ert.config._create_observation_dataframes import DEFAULT_LOCATION_RANGE_M


@settings(max_examples=10)
@given(summaries(summary_keys=st.just(["WOPR:OP1"])))
@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.integration_test
def test_reading_empty_summaries_raises(wopr_summary):
    smspec, unsmry = wopr_summary
    smspec.to_file("CASE.SMSPEC")
    unsmry.to_file("CASE.UNSMRY")
    with pytest.raises(InvalidResponseFile, match="Did not find any summary values"):
        SummaryConfig(
            name="summary", input_files=["CASE"], keys=["WWCT:OP1"]
        ).read_from_file(".", 0, 0)


def test_summary_config_normalizes_list_of_keys():
    assert SummaryConfig(
        name="summary", input_files=["CASE"], keys=["FOPR", "WOPR", "WOPR"]
    ).keys == [
        "FOPR",
        "WOPR",
    ]


@pytest.mark.usefixtures("use_tmpdir")
@given(st.binary(), st.binary())
def test_that_read_file_does_not_raise_unexpected_exceptions_on_invalid_file(
    smspec, unsmry
):
    Path("CASE.UNSMRY").write_bytes(unsmry)
    Path("CASE.SMSPEC").write_bytes(smspec)
    with suppress(InvalidResponseFile):
        SummaryConfig(
            name="summary", input_files=["CASE"], keys=["FOPR"]
        ).read_from_file(os.getcwd(), 1, 0)


def test_that_read_file_does_not_raise_unexpected_exceptions_on_missing_file(tmpdir):
    with pytest.raises(FileNotFoundError):
        SummaryConfig(
            name="summary", input_files=["NOT_CASE"], keys=["FOPR"]
        ).read_from_file(tmpdir, 1, 0)


@pytest.mark.usefixtures("use_tmpdir")
def test_that_read_file_does_not_raise_unexpected_exceptions_on_missing_directory(
    tmp_path,
):
    with pytest.raises(FileNotFoundError):
        SummaryConfig(
            name="summary", input_files=["CASE"], keys=["FOPR"]
        ).read_from_file(str(tmp_path / "DOES_NOT_EXIST"), 1, 0)


def create_summary_observation(loc_config_lines):
    config = dedent(
        """
    NUM_REALIZATIONS 3
    ECLBASE ECLIPSE_CASE_%d
    OBS_CONFIG observations
    GEN_KW KW_NAME template.txt kw.txt prior.txt
    RANDOM_SEED 1234
    """
    )
    obs_config = dedent(
        """
        SUMMARY_OBSERVATION FOPR_1
        {
        VALUE      = 0.9;
        ERROR      = 0.05;
        DATE       = 2014-09-10;
        KEY        = FOPR;
        """
        + loc_config_lines
        + """
        };
        """
    )
    Path("config.ert").write_text(config, encoding="utf-8")
    Path("observations").write_text(obs_config, encoding="utf-8")
    Path("template.txt").write_text("MY_KEYWORD <MY_KEYWORD>", encoding="utf-8")
    Path("prior.txt").write_text("MY_KEYWORD NORMAL 0 1", encoding="utf-8")

    ert_config = ErtConfig.from_file("config.ert")
    return ert_config.observations["summary"]


@pytest.mark.filterwarnings("ignore:Config contains a SUMMARY key but no forward model")
def test_that_summary_observations_can_be_instantiated_with_localization(
    tmpdir,
):
    with tmpdir.as_cwd():
        summary_observations = create_summary_observation(
            """
            LOCALIZATION {
                EAST   = 10;
                NORTH  = 10;
                RADIUS = 10;
            };
            """
        )
        assert all(
            loc_key in summary_observations.columns
            and summary_observations[loc_key][0] == 10
            for loc_key in ["location_x", "location_y", "location_range"]
        )
        assert len(summary_observations.columns) == 8


@pytest.mark.filterwarnings("ignore:Config contains a SUMMARY key but no forward model")
def test_that_summary_observations_without_radius_gets_defaulted(
    tmpdir,
):
    with tmpdir.as_cwd():
        summary_observations = create_summary_observation(
            """
            LOCALIZATION {
                EAST   = 10;
                NORTH  = 10;
            };
            """
        )
        assert "location_range" in summary_observations.columns
        assert summary_observations["location_range"][0] == DEFAULT_LOCATION_RANGE_M


@pytest.mark.parametrize(
    "loc_config_lines",
    [
        "EAST=10;\n",
        "NORTH=10;\n",
        "RADIUS=10;\n",
        "NORTH=10;\nRADIUS=10;\n",
        "EAST=10;\nRADIUS=10;\n",
    ],
)
@pytest.mark.filterwarnings("ignore:Config contains a SUMMARY key but no forward model")
def test_that_summary_observations_raises_error_when_east_or_north_are_undefined_given_localization_object(  # noqa: E501
    tmpdir,
    loc_config_lines,
):
    missing_keywords = {"NORTH", "EAST"} - set(
        re.findall(r"EAST|NORTH", loc_config_lines)
    )

    with pytest.raises(ConfigValidationError) as e, tmpdir.as_cwd():
        create_summary_observation("LOCALIZATION {" + loc_config_lines + "};")
    for kw in missing_keywords:
        assert f'Missing item "{kw}" in LOCALIZATION for FOPR_1' in str(e.value)


@pytest.mark.filterwarnings("ignore:Config contains a SUMMARY key but no forward model")
def test_that_summary_observations_raises_error_given_unknown_localization_key(
    tmpdir,
):
    with (
        pytest.raises(
            ConfigValidationError, match="Unknown FOO in LOCALIZATION for FOPR_1"
        ),
        tmpdir.as_cwd(),
    ):
        create_summary_observation("LOCALIZATION {FOO = BAZ;};")


@pytest.mark.usefixtures("copy_snake_oil_case")
def test_that_adding_one_localized_observation_to_snake_oil_case_can_be_internalized():
    obs_content = Path("observations/observations.txt").read_text(encoding="utf-8")
    obs_lines = obs_content.split("\n")
    observation_index = obs_lines.index("SUMMARY_OBSERVATION WOPR_OP1_36")
    localization_lines = [
        "LOCALIZATION {",
        "  EAST = 1;",
        "  NORTH = 2;",
        "  RADIUS = 3;",
        " };",
    ]
    for i, line in enumerate(localization_lines):
        obs_lines.insert(observation_index + 2 + i, line)
    new_obs_content = "\n".join(obs_lines)
    Path("observations/observations.txt").write_text(new_obs_content, encoding="utf-8")
    summary = ErtConfig.from_file("snake_oil.ert").observations["summary"]
    assert summary["location_x"].dtype == pl.Float32
    assert summary["location_y"].dtype == pl.Float32
    assert summary["location_range"].dtype == pl.Float32

    localized_entry = summary.filter(
        pl.col("observation_key").str.contains("WOPR_OP1_36")
    )
    assert localized_entry["location_x"].to_list() == [1]
    assert localized_entry["location_y"].to_list() == [2]
    assert localized_entry["location_range"].to_list() == [3]


def test_that_defaulted_summary_obs_values_have_type_float32(tmpdir):
    with tmpdir.as_cwd():
        summary_observations = create_summary_observation(loc_config_lines="")
    assert summary_observations["location_x"].dtype == pl.Float32
    assert summary_observations["location_y"].dtype == pl.Float32
    assert summary_observations["location_range"].dtype == pl.Float32
