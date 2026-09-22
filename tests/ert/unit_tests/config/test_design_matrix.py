from unittest.mock import patch

import pytest

from ert.config import (
    ConfigValidationError,
    DesignMatrix,
    GenKwConfig,
    LocalizationType,
)
from ert.config.distribution import RawSettings
from ert.config.gen_kw_config import DataSource


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
