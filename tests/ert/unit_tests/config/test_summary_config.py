import os
import re
from contextlib import suppress
from pathlib import Path
from textwrap import dedent

import hypothesis.strategies as st
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


def test_that_summary_observations_can_be_instantiated_with_localization(
    tmpdir,
):
    with tmpdir.as_cwd():
        summary_observations = create_summary_observation(
            """
            LOCATION_X=10;
            LOCATION_Y=10;
            LOCATION_RANGE=10;
            """
        )
        assert all(
            loc_key in summary_observations.columns
            and summary_observations[loc_key][0] == 10
            for loc_key in ["location_x", "location_y", "location_range"]
        )
        assert len(summary_observations.columns) == 8


def test_that_summary_observations_without_location_range_gets_defaulted(
    tmpdir,
):
    with tmpdir.as_cwd():
        summary_observations = create_summary_observation(
            """
            LOCATION_X=10;
            LOCATION_Y=10;
            """
        )
        assert "location_range" in summary_observations.columns
        assert summary_observations["location_range"][0] == DEFAULT_LOCATION_RANGE_M


@pytest.mark.parametrize(
    "loc_config_lines",
    [
        "LOCATION_X=10;\n",
        "LOCATION_Y=10;\n",
        "LOCATION_RANGE=10;\n",
        "LOCATION_Y=10;\nLOCATION_RANGE=10;\n",
        "LOCATION_X=10;\nLOCATION_RANGE=10;\n",
    ],
)
def test_that_summary_observations_raises_config_validation_error_when_loc_x_or_loc_y_are_undefined_given_any_loc_key(  # noqa: E501
    tmpdir,
    loc_config_lines,
):
    location_pattern = r"LOCATION_[a-zA-Z]*"
    matches = re.findall(location_pattern, loc_config_lines)
    expected_err_msgs = [
        "Localization for observation FOPR_1 is misconfigured.",
        f"Only {', '.join(matches)} were provided.",
        "ensure that both LOCATION_X and LOCATION_Y are defined",
    ]
    with pytest.raises(ConfigValidationError) as e, tmpdir.as_cwd():
        create_summary_observation(loc_config_lines)
    for msg in expected_err_msgs:
        assert msg in str(e.value)
