from unittest.mock import patch

import polars as pl
import pytest

from ert.config import (
    ConfigValidationError,
    ConfigWarning,
    DesignMatrix,
    GenKwConfig,
    LocalizationType,
)
from ert.config.distribution import RawSettings
from ert.config.gen_kw_config import DataSource
from tests.ert.conftest import _create_design_matrix


def test_that_categorical_design_matrix_parameters_are_excluded_from_update(tmp_path):
    design_path = tmp_path / "design_matrix.xlsx"
    design_matrix_df = pl.DataFrame(
        {
            "REAL": [0, 1, 2],
            "a": [1.0, 2.0, 3.0],
            "b": ["low", "medium", "high"],
        }
    )
    _create_design_matrix(design_path, design_matrix_df)
    dm = DesignMatrix(
        filename=design_path,
        design_sheet="DesignSheet",
        default_sheet=None,
        update=True,
        update_strategy=LocalizationType.GLOBAL,
    )

    with pytest.warns(ConfigWarning, match="categorical values.*: b"):
        merged_params = {
            p.name: p for p in dm.merge_with_existing_parameters(existing_parameters=[])
        }

    assert merged_params["a"].update_strategy == LocalizationType.GLOBAL
    assert merged_params["b"].update_strategy is None


def test_that_categorical_design_matrix_parameters_overlapping_gen_kw_are_excluded_from_update(  # ruff: ignore[line-too-long]
    tmp_path,
):
    design_path = tmp_path / "design_matrix.xlsx"
    design_matrix_df = pl.DataFrame(
        {
            "REAL": [0, 1, 2],
            "a": [1.0, 2.0, 3.0],
            "b": ["low", "medium", "high"],
        }
    )
    _create_design_matrix(design_path, design_matrix_df)
    dm = DesignMatrix(
        filename=design_path,
        design_sheet="DesignSheet",
        default_sheet=None,
        update=True,
        update_strategy=LocalizationType.GLOBAL,
    )

    # Existing GEN_KW parameters with the same names, overridden by the design
    # matrix (default priority), exercising the existing-parameter merge branch.
    existing_parameters = [
        GenKwConfig(
            name="a",
            distribution=RawSettings(name="raw"),
            update_strategy=LocalizationType.ADAPTIVE,
        ),
        GenKwConfig(
            name="b",
            distribution=RawSettings(name="raw"),
            update_strategy=LocalizationType.ADAPTIVE,
        ),
    ]

    with pytest.warns(ConfigWarning, match="categorical values.*: b"):
        merged_params = {
            p.name: p
            for p in dm.merge_with_existing_parameters(
                existing_parameters=existing_parameters
            )
        }

    assert merged_params["a"].update_strategy == LocalizationType.GLOBAL
    assert merged_params["b"].update_strategy is None


def test_that_categorical_parameters_from_a_merged_design_matrix_are_excluded_from_update(  # ruff: ignore[line-too-long]
    tmp_path,
):
    design_path_1 = tmp_path / "design_matrix_1.xlsx"
    _create_design_matrix(
        design_path_1,
        pl.DataFrame(
            {
                "REAL": [0, 1, 2],
                "a": [1.0, 2.0, 3.0],
            }
        ),
    )
    dm1 = DesignMatrix(
        filename=design_path_1,
        design_sheet="DesignSheet",
        default_sheet=None,
        update=True,
        update_strategy=LocalizationType.GLOBAL,
    )

    design_path_2 = tmp_path / "design_matrix_2.xlsx"
    _create_design_matrix(
        design_path_2,
        pl.DataFrame(
            {
                "REAL": [0, 1, 2],
                "b": ["low", "medium", "high"],
            }
        ),
    )
    dm2 = DesignMatrix(
        filename=design_path_2,
        design_sheet="DesignSheet",
        default_sheet=None,
    )

    dm1.merge_with_other(dm2)

    with pytest.warns(ConfigWarning, match="categorical values.*: b"):
        merged_params = {
            p.name: p
            for p in dm1.merge_with_existing_parameters(existing_parameters=[])
        }

    assert merged_params["a"].update_strategy == LocalizationType.GLOBAL
    assert merged_params["b"].update_strategy is None


@pytest.mark.parametrize(
    ("update", "expected"),
    [("TRUE", True), ("FALSE", False), ("TrUe", True), ("FaLSE", False)],
)
def test_that_from_config_list_with_update_option_parses_boolean_value(
    update, expected
):
    config_list = [
        "dummy.xlsx",
        {
            "DESIGN_SHEET": "DesignSheet",
            "DEFAULT_SHEET": "DefaultSheet",
            "PRIORITY": "design_matrix",
            "UPDATE": update,
        },
    ]

    with patch.object(DesignMatrix, "__post_init__", return_value=None):
        dm = DesignMatrix.from_config_list(
            config_list,
            update_strategy=None,
        )
        assert dm.update is expected


def test_that_from_config_list_with_invalid_update_option_throws():
    config_list = [
        "dummy.xlsx",
        {
            "DESIGN_SHEET": "DesignSheet",
            "DEFAULT_SHEET": "DefaultSheet",
            "PRIORITY": "design_matrix",
            "UPDATE": "INVALID",
        },
    ]

    with (
        patch.object(DesignMatrix, "__post_init__", return_value=None),
        pytest.raises(
            ConfigValidationError,
            match="UPDATE must be either 'TRUE' or 'FALSE'; is 'INVALID'",
        ),
    ):
        DesignMatrix.from_config_list(config_list, update_strategy=None)


@pytest.mark.parametrize("priority", ["design_matrix", "sampled"])
def test_that_merge_with_existing_parameters_merges_correctly_with_no_existing_params_and_no_update(  # ruff: ignore[line-too-long]
    priority,
):

    with patch.object(DesignMatrix, "__post_init__", return_value=None):
        dm = DesignMatrix.from_config_list(
            [
                "dummy.xlsx",
                {
                    "DESIGN_SHEET": "DesignSheet",
                    "DEFAULT_SHEET": "DefaultSheet",
                    "PRIORITY": priority,
                },
            ],
            update_strategy=LocalizationType.ADAPTIVE,
        )

    # state after loading design matrix
    dm.parameter_configurations = [
        GenKwConfig(
            name="param1",
            distribution=RawSettings(name="raw"),
            update_strategy=None,
        ),
        GenKwConfig(
            name="param2",
            distribution=RawSettings(name="raw"),
            update_strategy=None,
        ),
        GenKwConfig(
            name="param3",
            distribution=RawSettings(name="raw"),
            update_strategy=None,
        ),
    ]

    merged_params = dm.merge_with_existing_parameters(existing_parameters=[])

    assert merged_params == dm.parameter_configurations


@pytest.mark.parametrize("priority", ["design_matrix", "sampled"])
def test_that_merge_with_existing_parameters_merges_correctly_with_no_existing_params_and_update_true(  # ruff: ignore[line-too-long]
    priority,
):

    with patch.object(DesignMatrix, "__post_init__", return_value=None):
        dm = DesignMatrix.from_config_list(
            [
                "dummy.xlsx",
                {
                    "DESIGN_SHEET": "DesignSheet",
                    "DEFAULT_SHEET": "DefaultSheet",
                    "PRIORITY": priority,
                    "UPDATE": "TRUE",
                },
            ],
            update_strategy=LocalizationType.ADAPTIVE,
        )

    # state after loading design matrix
    dm.parameter_configurations = [
        GenKwConfig(
            name="param1",
            distribution=RawSettings(name="raw"),
            update_strategy=None,
        ),
        GenKwConfig(
            name="param2",
            distribution=RawSettings(name="raw"),
            update_strategy=None,
        ),
        GenKwConfig(
            name="param3",
            distribution=RawSettings(name="raw"),
            update_strategy=None,
        ),
    ]

    merged_params = dm.merge_with_existing_parameters(existing_parameters=[])

    for param in merged_params:
        assert param.update_strategy == LocalizationType.ADAPTIVE


@pytest.mark.parametrize("priority", ["design_matrix", "sampled"])
def test_that_merge_with_existing_parameter_with_update_true_and_no_parameter_update_strategy_sets_update_to_global(  # ruff: ignore[line-too-long]
    priority,
):
    with patch.object(DesignMatrix, "__post_init__", return_value=None):
        dm = DesignMatrix.from_config_list(
            [
                "dummy.xlsx",
                {
                    "DESIGN_SHEET": "DesignSheet",
                    "DEFAULT_SHEET": "DefaultSheet",
                    "PRIORITY": priority,
                    "UPDATE": "TRUE",
                },
            ],
            update_strategy=None,
        )

    # state after loading design matrix
    dm.parameter_configurations = [
        GenKwConfig(
            name="param1",
            distribution=RawSettings(name="raw"),
            input_source=DataSource.DESIGN_MATRIX,
            update_strategy=None,
        ),
        GenKwConfig(
            name="param2",
            distribution=RawSettings(name="raw"),
            input_source=DataSource.DESIGN_MATRIX,
            update_strategy=None,
        ),
        GenKwConfig(
            name="param3",
            distribution=RawSettings(name="raw"),
            input_source=DataSource.DESIGN_MATRIX,
            update_strategy=None,
        ),
    ]

    dm.parameter_priority = {
        cfg.name: dm.priority_source for cfg in dm.parameter_configurations
    }

    new_params = dm.merge_with_existing_parameters(existing_parameters=[])
    for param in new_params:
        assert param.update_strategy == LocalizationType.GLOBAL


@pytest.mark.parametrize(
    ("update", "global_update_strategy", "existing_parameters", "expected"),
    [
        pytest.param(
            "FALSE",
            LocalizationType.ADAPTIVE,
            [
                GenKwConfig(
                    name="param2",
                    distribution=RawSettings(name="raw"),
                    update_strategy=LocalizationType.GLOBAL,
                    input_source=DataSource.SAMPLED,
                ),
                GenKwConfig(
                    name="param4",
                    distribution=RawSettings(name="raw"),
                    update_strategy=LocalizationType.GLOBAL,
                    input_source=DataSource.SAMPLED,
                ),
            ],
            {
                "param1": None,
                "param2": LocalizationType.GLOBAL,
                "param3": None,
                "param4": LocalizationType.GLOBAL,
            },
            id="update_false_leaves_design_matrix_params_unset",
        ),
        pytest.param(
            "TRUE",
            LocalizationType.DISTANCE,
            [
                GenKwConfig(
                    name="param1",
                    distribution=RawSettings(name="raw"),
                    update_strategy=LocalizationType.ADAPTIVE,
                    input_source=DataSource.SAMPLED,
                ),
                GenKwConfig(
                    name="param2",
                    distribution=RawSettings(name="raw"),
                    update_strategy=LocalizationType.ADAPTIVE,
                    input_source=DataSource.SAMPLED,
                ),
                GenKwConfig(
                    name="param4",
                    distribution=RawSettings(name="raw"),
                    update_strategy=LocalizationType.ADAPTIVE,
                    input_source=DataSource.SAMPLED,
                ),
            ],
            {
                "param1": LocalizationType.DISTANCE,
                "param2": LocalizationType.ADAPTIVE,
                "param3": LocalizationType.DISTANCE,
                "param4": LocalizationType.ADAPTIVE,
            },
            id="update_true_overrides_by_priority_and_keeps_sampled_priority_params",
        ),
    ],
)
def test_that_merge_with_existing_parameters_respects_update_flag_and_priority(
    update, global_update_strategy, existing_parameters, expected
):
    with patch.object(DesignMatrix, "__post_init__", return_value=None):
        dm = DesignMatrix.from_config_list(
            [
                "dummy.xlsx",
                {
                    "DESIGN_SHEET": "DesignSheet",
                    "DEFAULT_SHEET": "DefaultSheet",
                    "PRIORITY": "design_matrix",
                    "UPDATE": update,
                },
            ],
            update_strategy=global_update_strategy,
        )

    # state after loading design matrix
    dm.parameter_configurations = [
        GenKwConfig(
            name="param1",
            distribution=RawSettings(name="raw"),
            input_source=DataSource.DESIGN_MATRIX,
            update_strategy=None,
        ),
        GenKwConfig(
            name="param2",
            distribution=RawSettings(name="raw"),
            input_source=DataSource.DESIGN_MATRIX,
            update_strategy=None,
        ),
        GenKwConfig(
            name="param3",
            distribution=RawSettings(name="raw"),
            input_source=DataSource.DESIGN_MATRIX,
            update_strategy=None,
        ),
    ]

    dm.parameter_priority = {
        "param1": DataSource.DESIGN_MATRIX.value,
        "param2": DataSource.SAMPLED.value,
    }

    merged_params = dm.merge_with_existing_parameters(
        existing_parameters=existing_parameters
    )

    assert len(merged_params) == len(expected)
    for name, strategy in expected.items():
        assert any(
            cfg.name == name and cfg.update_strategy is strategy
            for cfg in merged_params
        )
