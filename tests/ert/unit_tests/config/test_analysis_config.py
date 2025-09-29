import logging
import os.path
from textwrap import dedent

import hypothesis.strategies as st
import pandas as pd
import pytest
from hypothesis import given

from ert.config import (
    AnalysisConfig,
    ConfigValidationError,
    ErtConfig,
    ESSettings,
)
from ert.config.parsing import ConfigKeys, ConfigWarning


@pytest.mark.integration_test
def test_analysis_config_from_file_is_same_as_from_dict(monkeypatch, tmp_path):
    with pd.ExcelWriter(tmp_path / "my_design_matrix.xlsx") as xl_write:
        design_matrix_df = pd.DataFrame(
            {
                "REAL": [0, 1, 2],
                "a": [1, 2, 3],
                "b": [0, 2, 0],
            }
        )
        default_sheet_df = pd.DataFrame([["a", 1], ["b", 4]])
        design_matrix_df.to_excel(xl_write, index=False, sheet_name="my_sheet")
        default_sheet_df.to_excel(
            xl_write, index=False, sheet_name="my_default_sheet", header=False
        )
    monkeypatch.chdir(tmp_path)
    assert ErtConfig.from_file_contents(
        dedent(
            """
                NUM_REALIZATIONS 3
                MIN_REALIZATIONS 3
                ANALYSIS_SET_VAR STD_ENKF ENKF_TRUNCATION 0.8
                DESIGN_MATRIX my_design_matrix.xlsx DESIGN_SHEET:my_sheet \
                    DEFAULT_SHEET:my_default_sheet
                """
        )
    ).analysis_config == AnalysisConfig.from_dict(
        {
            ConfigKeys.NUM_REALIZATIONS: 3,
            ConfigKeys.MIN_REALIZATIONS: "3",
            ConfigKeys.ANALYSIS_SET_VAR: [
                ("STD_ENKF", "ENKF_TRUNCATION", 0.8),
            ],
            ConfigKeys.DESIGN_MATRIX: [
                [
                    os.path.abspath("my_design_matrix.xlsx"),
                    {
                        "DESIGN_SHEET": "my_sheet",
                        "DEFAULT_SHEET": "my_default_sheet",
                    },
                ]
            ],
        }
    )


def test_merging_ignores_identical_design_matrices(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.WARNING)
    with pd.ExcelWriter(tmp_path / "my_design_matrix.xlsx") as xl_write:
        design_matrix_df = pd.DataFrame(
            {
                "REAL": [0, 1, 2],
                "a": [1, 2, 3],
                "b": [0, 2, 0],
            }
        )
        default_sheet_df = pd.DataFrame()
        design_matrix_df.to_excel(xl_write, index=False, sheet_name="my_sheet")
        default_sheet_df.to_excel(
            xl_write, index=False, sheet_name="my_default_sheet", header=False
        )
    monkeypatch.chdir(tmp_path)
    AnalysisConfig.from_dict(
        {
            ConfigKeys.NUM_REALIZATIONS: 3,
            ConfigKeys.DESIGN_MATRIX: [
                [
                    "my_design_matrix.xlsx",
                    {
                        "DESIGN_SHEET": "my_sheet",
                        "DEFAULT_SHEET": "my_default_sheet",
                    },
                ],
                [
                    "my_design_matrix.xlsx",
                    {
                        "DESIGN_SHEET": "my_sheet",
                        "DEFAULT_SHEET": "my_default_sheet",
                    },
                ],
            ],
        }
    )
    assert (
        "Duplicate DESIGN_MATRIX entries "
        "DesignMatrix(xls_filename="
        "PosixPath('my_design_matrix.xlsx'), "
        "design_sheet='my_sheet', "
        "default_sheet='my_default_sheet'), only reading once." in caplog.text
    )


@pytest.mark.filterwarnings(
    "ignore:.*MIN_REALIZATIONS set to more "
    "than NUM_REALIZATIONS.*:ert.config.ConfigWarning"
)
@pytest.mark.parametrize(
    "num_realization, min_realizations, expected_min_real",
    [
        (80, "10%", 8),
        (5, "2%", 1),
        (8, "50%", 4),
        (8, "100%", 8),
        (8, "100%", 8),
        (8, "10%", 1),
        (80, "0", 80),
        (80, "0", 80),
        (900, "10", 10),
        (80, "50", 50),
        (80, "80", 80),
        (80, "100", 80),
    ],
)
def test_analysis_config_min_realizations(
    num_realization, min_realizations, expected_min_real
):
    assert (
        AnalysisConfig.from_dict(
            {
                ConfigKeys.NUM_REALIZATIONS: num_realization,
                ConfigKeys.MIN_REALIZATIONS: min_realizations,
            }
        ).minimum_required_realizations
        == expected_min_real
    )


def test_invalid_min_realization_raises_config_validation_error():
    with pytest.raises(
        ConfigValidationError, match="MIN_REALIZATIONS value is not integer"
    ):
        AnalysisConfig.from_dict(
            {
                ConfigKeys.NUM_REALIZATIONS: 1,
                ConfigKeys.MIN_REALIZATIONS: "1s",
            }
        )


def test_invalid_design_matrix_format_raises_validation_error():
    with pytest.raises(
        ConfigValidationError,
        match="DESIGN_MATRIX must be of format \\.xls or \\.xlsx; is 'my_matrix\\.txt'",
    ):
        AnalysisConfig.from_dict(
            {
                ConfigKeys.NUM_REALIZATIONS: 1,
                ConfigKeys.DESIGN_MATRIX: [
                    [
                        "my_matrix.txt",
                        {
                            "DESIGN_SHEET": "sheet1",
                            "DEFAULT_SHEET": "sheet2",
                        },
                    ],
                ],
            }
        )


def test_invalid_min_realization_percentage_raises_config_validation_error():
    with pytest.raises(
        ConfigValidationError,
        match="MIN_REALIZATIONS 'd%s' contained % but was not a valid percentage",
    ):
        AnalysisConfig.from_dict(
            {
                ConfigKeys.NUM_REALIZATIONS: 1,
                ConfigKeys.MIN_REALIZATIONS: "d%s",
            }
        )


@pytest.mark.parametrize(
    "value, expected", [("100%", 50), ("+34", 34), ("-1", -1), ("50.5%", 26)]
)
def test_valid_min_realization(value, expected):
    assert (
        AnalysisConfig.from_dict(
            {
                ConfigKeys.NUM_REALIZATIONS: 50,
                ConfigKeys.MIN_REALIZATIONS: value,
            }
        ).minimum_required_realizations
        == expected
    )


@pytest.mark.parametrize(
    "analysis_config", [AnalysisConfig(), AnalysisConfig.from_dict({})]
)
def test_analysis_config_modules(analysis_config):
    assert isinstance(analysis_config.es_settings, ESSettings)


def test_unknown_variable_raises_validation_error():
    with pytest.raises(ConfigValidationError, match="Extra inputs are not permitted"):
        _ = AnalysisConfig.from_dict(
            {
                ConfigKeys.ANALYSIS_SET_VAR: [["STD_ENKF", "BAR", "1"]],
            }
        )


def test_default_alpha_is_set():
    default_alpha = 3.0
    assert (
        AnalysisConfig.from_dict({}).observation_settings.outlier_settings.alpha
        == default_alpha
    )


@given(st.floats(allow_nan=False, allow_infinity=False))
def test_alpha_is_set_from_corresponding_key(value):
    assert (
        AnalysisConfig.from_dict(
            {ConfigKeys.ENKF_ALPHA: value}
        ).observation_settings.outlier_settings.alpha
        == value
    )


def test_default_std_cutoff_is_set():
    default_std_cutoff = 1e-6
    assert (
        AnalysisConfig.from_dict({}).observation_settings.outlier_settings.std_cutoff
        == default_std_cutoff
    )


@given(st.floats(allow_nan=False, allow_infinity=False, min_value=1e-12))
def test_std_cutoff_is_set_from_corresponding_key(value):
    assert (
        AnalysisConfig.from_dict(
            {ConfigKeys.STD_CUTOFF: value}
        ).observation_settings.outlier_settings.std_cutoff
        == value
    )


@given(st.integers(min_value=1))
def test_that_the_default_min_realizations_is_all_realizations(value):
    assert (
        AnalysisConfig.from_dict(
            {ConfigKeys.NUM_REALIZATIONS: value}
        ).minimum_required_realizations
        == value
    )


@given(st.integers(min_value=1))
def test_min_realization_is_set_from_corresponding_keyword(value):
    assert (
        AnalysisConfig.from_dict(
            {
                ConfigKeys.NUM_REALIZATIONS: value + 1,
                ConfigKeys.MIN_REALIZATIONS: str(value),
            }
        ).minimum_required_realizations
        == value
    )
    assert (
        AnalysisConfig(
            minimum_required_realizations=value
        ).minimum_required_realizations
        == value
    )


def test_giving_larger_min_than_num_realizations_warns():
    with pytest.warns(
        ConfigWarning, match="MIN_REALIZATIONS is set to more than NUM_REALIZATIONS"
    ):
        _ = AnalysisConfig.from_dict(
            {
                ConfigKeys.NUM_REALIZATIONS: 1,
                ConfigKeys.MIN_REALIZATIONS: "2",
            }
        )


def test_num_realizations_0_means_all():
    """For legacy reasons, `NUM_REALIZATIONS 0` means all must pass."""
    assert (
        AnalysisConfig.from_dict(
            {
                ConfigKeys.NUM_REALIZATIONS: 100,
                ConfigKeys.MIN_REALIZATIONS: "0",
            }
        ).minimum_required_realizations
        == 100
    )


@pytest.mark.parametrize(
    "config, expected",
    [
        ("exact", "EXACT"),
        ("0", "EXACT"),
        ("subspace", "SUBSPACE"),
        ("1", "SUBSPACE"),
        ("subspace_exact_r", "SUBSPACE"),
        ("SUBSPACE_EXACT_R", "SUBSPACE"),
    ],
)
def test_incorrect_variable_deprecation_warning(config, expected):
    with pytest.warns(
        match=(
            f"Using {config} is deprecated, use:\n"
            f"ANALYSIS_SET_VAR STD_ENKF INVERSION {expected}"
        )
    ):
        AnalysisConfig.from_dict(
            {
                ConfigKeys.ANALYSIS_SET_VAR: [["STD_ENKF", "INVERSION", config]],
            }
        )


@pytest.mark.parametrize(
    "config",
    [
        ("2"),
        ("3"),
        ("SUBSPACE_RE"),
        ("SUBSPACE_EE_R"),
        ("subspace_re"),
        ("subspace_ee_r"),
    ],
)
def test_incorrect_variable_inversion(config):
    with pytest.raises(
        ConfigValidationError,
        match=f"Input should be 'EXACT' or 'SUBSPACE'.*{config.upper()}",
    ):
        AnalysisConfig.from_dict(
            {
                ConfigKeys.ANALYSIS_SET_VAR: [["STD_ENKF", "INVERSION", config]],
            }
        )


@pytest.mark.parametrize(
    "config, expected",
    [
        ([["OBSERVATIONS", "AUTO_SCALE", "OBS_*"]], [["OBS_*"]]),
        ([["OBSERVATIONS", "AUTO_SCALE", "ONE,TWO"]], [["ONE", "TWO"]]),
        (
            [
                ["OBSERVATIONS", "AUTO_SCALE", "OBS_*"],
                ["OBSERVATIONS", "AUTO_SCALE", "SINGLE"],
            ],
            [["OBS_*"], ["SINGLE"]],
        ),
    ],
)
def test_misfit_configuration(config, expected):
    analysis_config = AnalysisConfig.from_dict(
        {
            ConfigKeys.ANALYSIS_SET_VAR: config,
        }
    )
    assert analysis_config.observation_settings.auto_scale_observations == expected


@pytest.mark.parametrize(
    "config, expectation",
    [
        (
            [["OBSERVATIONS", "SAUTO_SCALE", "OBS_*"]],
            pytest.raises(ConfigValidationError, match="Unknown variable"),
        ),
        (
            [["NOT_A_THING", "AUTO_SCALE", "OBS_*"]],
            pytest.raises(ConfigValidationError, match="ANALYSIS_SET_VAR NOT_A_THING"),
        ),
    ],
)
def test_config_wrong_module(config, expectation):
    with expectation:
        AnalysisConfig.from_dict(
            {
                ConfigKeys.ANALYSIS_SET_VAR: config,
            }
        )
