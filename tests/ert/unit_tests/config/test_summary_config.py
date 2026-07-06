import re
from contextlib import suppress
from datetime import datetime
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
    summary_config,
)
from ert.config._create_observation_dataframes import create_observation_dataframes
from ert.config._observations import DEFAULT_LOCALIZATION_RADIUS
from ert.warnings import PostExperimentWarning


@settings(max_examples=10)
@given(summaries(summary_keys=st.just(["WOPR:OP1"])))
@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.slow
@pytest.mark.filterwarnings(r"ignore:Could not find responses for key\(s\)")
def test_that_reading_empty_summaries_returns_empty_df_with_column_schema(wopr_summary):
    smspec, unsmry = wopr_summary
    smspec.to_file("CASE.SMSPEC")
    unsmry.to_file("CASE.UNSMRY")
    responses_df = SummaryConfig(
        input_files=["CASE"], keys=["WWCT:OP1"]
    ).read_from_file(".", 0, 0)
    assert responses_df.is_empty()
    assert responses_df.columns == ["response_key", "time", "values"]


def test_summary_config_normalizes_list_of_keys():
    assert SummaryConfig(input_files=["CASE"], keys=["FOPR", "WOPR", "WOPR"]).keys == [
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
        SummaryConfig(input_files=["CASE"], keys=["FOPR"]).read_from_file(
            Path.cwd(), 1, 0
        )


def test_that_read_file_does_not_raise_unexpected_exceptions_on_missing_file(tmpdir):
    with pytest.raises(FileNotFoundError):
        SummaryConfig(input_files=["NOT_CASE"], keys=["FOPR"]).read_from_file(
            tmpdir, 1, 0
        )


@pytest.mark.usefixtures("use_tmpdir")
def test_that_read_file_does_not_raise_unexpected_exceptions_on_missing_directory(
    tmp_path,
):
    with pytest.raises(FileNotFoundError):
        SummaryConfig(input_files=["CASE"], keys=["FOPR"]).read_from_file(
            str(tmp_path / "DOES_NOT_EXIST"), 1, 0
        )


def create_observations(config, obs_config):
    Path("config.ert").write_text(config, encoding="utf-8")
    Path("observations").write_text(obs_config, encoding="utf-8")
    Path("template.txt").write_text("MY_KEYWORD <MY_KEYWORD>", encoding="utf-8")
    Path("prior.txt").write_text("MY_KEYWORD NORMAL 0 1", encoding="utf-8")

    ert_config = ErtConfig.from_file("config.ert")
    return create_observation_dataframes(
        ert_config.observation_declarations, ert_config.shape_registry
    )


def create_localization_summary_observation(loc_config_lines):
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
    return create_observations(config, obs_config)["summary"]


@pytest.mark.filterwarnings("ignore:Config contains a SUMMARY key but no forward model")
def test_that_summary_observations_can_be_instantiated_with_localization(
    tmpdir,
):
    with tmpdir.as_cwd():
        summary_observations = create_localization_summary_observation(
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
            for loc_key in ["east", "north", "radius"]
        )
        assert len(summary_observations.columns) == 8


@pytest.mark.filterwarnings("ignore:Config contains a SUMMARY key but no forward model")
def test_that_summary_observations_without_radius_gets_defaulted(
    tmpdir,
):
    with tmpdir.as_cwd():
        summary_observations = create_localization_summary_observation(
            """
            LOCALIZATION {
                EAST   = 10;
                NORTH  = 10;
            };
            """
        )
        assert "radius" in summary_observations.columns
        assert summary_observations["radius"][0] == DEFAULT_LOCALIZATION_RADIUS


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
        create_localization_summary_observation(
            "LOCALIZATION {" + loc_config_lines + "};"
        )
    for kw in missing_keywords:
        assert f'Missing item "{kw}" in LOCALIZATION for SUMMARY_OBSERVATION' in str(
            e.value
        )


@pytest.mark.filterwarnings("ignore:Config contains a SUMMARY key but no forward model")
def test_that_summary_observations_raises_error_given_unknown_localization_key(
    tmpdir,
):
    with (
        pytest.raises(
            ConfigValidationError,
            match=r"Unknown key 'FOO' in LOCALIZATION for SUMMARY_OBSERVATION",
        ),
        tmpdir.as_cwd(),
    ):
        create_localization_summary_observation("LOCALIZATION {FOO = BAZ;};")


@pytest.mark.usefixtures("copy_snake_oil_case")
@pytest.mark.filterwarnings("ignore:Config contains a SUMMARY key but no forward model")
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
    ert_config = ErtConfig.from_file("snake_oil.ert")
    summary = create_observation_dataframes(
        ert_config.observation_declarations, ert_config.shape_registry
    )["summary"]
    assert summary["east"].dtype == pl.Float32
    assert summary["north"].dtype == pl.Float32
    assert summary["radius"].dtype == pl.Float32

    localized_entry = summary.filter(
        pl.col("observation_key").str.contains("WOPR_OP1_36")
    )
    assert localized_entry["east"].to_list() == [1]
    assert localized_entry["north"].to_list() == [2]
    assert localized_entry["radius"].to_list() == [3]


@pytest.mark.filterwarnings("ignore:Config contains a SUMMARY key but no forward model")
def test_that_defaulted_summary_obs_values_have_type_float32(tmpdir):
    with tmpdir.as_cwd():
        summary_observations = create_localization_summary_observation(
            loc_config_lines=""
        )
    assert summary_observations["east"].dtype == pl.Float32
    assert summary_observations["north"].dtype == pl.Float32
    assert summary_observations["radius"].dtype == pl.Float32


def test_that_when_not_finding_response_obs_keys_raises_warning(monkeypatch):
    response_keys = ["WOPR:OP1", "FOPR"]
    obs_keys = ["WOPR:OP1", "WWCT:OP1", "WOPR:OP2"]

    def mock_read_summary(*args):
        return (
            None,
            response_keys,
            [datetime.fromisoformat("2012-10-10")] * 2,
            [[None] * 2] * 2,
        )

    monkeypatch.setattr(summary_config, "read_summary", mock_read_summary)

    with pytest.warns(PostExperimentWarning) as warnings:
        SummaryConfig(input_files=["CASE"], keys=list(obs_keys)).read_from_file(
            ".", 0, 0
        )

    warning = next(
        str(w.message) for w in warnings if "Could not find response" in str(w.message)
    )
    expected_warning = dedent(
        """\
        Could not find responses for key(s) in 'CASE':
        WOPR:OP2
        WWCT:OP1"""
    )
    assert warning == expected_warning
