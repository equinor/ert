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


def _assert_parameter_configurations(list1, list2):
    assert len(list1) == len(list2)
    list1 = sorted(list1, key=lambda cfg: cfg.name)
    list2 = sorted(list2, key=lambda cfg: cfg.name)
    for cfg1, cfg2 in zip(list1, list2, strict=True):
        assert cfg1.name == cfg2.name
        assert cfg1.update_strategy == cfg2.update_strategy


@pytest.mark.parametrize(
    ("update", "expected"),
    [("TRUE", True), ("FALSE", False), ("TrUe", True), ("FaLSE", False)],
)
def test_that_from_config_list_with_update_option_sets_update_correctly(
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
            config_list, parameter_type_update_strategies={}
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
        DesignMatrix.from_config_list(config_list, parameter_type_update_strategies={})


def test_that_merge_with_existing_parameters_merges_correctly_with_no_existing_params():

    with patch.object(DesignMatrix, "__post_init__", return_value=None):
        dm = DesignMatrix.from_config_list(
            [
                "dummy.xlsx",
                {
                    "DESIGN_SHEET": "DesignSheet",
                    "DEFAULT_SHEET": "DefaultSheet",
                    "PRIORITY": "design_matrix",
                },
            ],
            parameter_type_update_strategies={"GEN_KW": LocalizationType.ADAPTIVE},
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


def test_that_merge_with_existing_parameters_merges_correctly_with_no_existing_params_and_update_true():  # ruff: ignore[line-too-long]

    with patch.object(DesignMatrix, "__post_init__", return_value=None):
        dm = DesignMatrix.from_config_list(
            [
                "dummy.xlsx",
                {
                    "DESIGN_SHEET": "DesignSheet",
                    "DEFAULT_SHEET": "DefaultSheet",
                    "PRIORITY": "design_matrix",
                    "UPDATE": "TRUE",
                },
            ],
            parameter_type_update_strategies={"GEN_KW": LocalizationType.ADAPTIVE},
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

    assert merged_params == [
        GenKwConfig(
            name="param1",
            distribution=RawSettings(name="raw"),
            update_strategy=LocalizationType.ADAPTIVE,
        ),
        GenKwConfig(
            name="param2",
            distribution=RawSettings(name="raw"),
            update_strategy=LocalizationType.ADAPTIVE,
        ),
        GenKwConfig(
            name="param3",
            distribution=RawSettings(name="raw"),
            update_strategy=LocalizationType.ADAPTIVE,
        ),
    ]


def test_that_merge_with_existing_parameters_merges_correctly_with_existing_params_update_false():  # ruff: ignore[line-too-long]

    with patch.object(DesignMatrix, "__post_init__", return_value=None):
        dm = DesignMatrix.from_config_list(
            [
                "dummy.xlsx",
                {
                    "DESIGN_SHEET": "DesignSheet",
                    "DEFAULT_SHEET": "DefaultSheet",
                    "PRIORITY": "design_matrix",
                    "UPDATE": "FALSE",
                },
            ],
            parameter_type_update_strategies={"GEN_KW": LocalizationType.ADAPTIVE},
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

    merged_params = dm.merge_with_existing_parameters(
        existing_parameters=[
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
        ]
    )

    _assert_parameter_configurations(
        merged_params,
        [
            *dm.parameter_configurations,
            GenKwConfig(
                name="param4",
                distribution=RawSettings(name="raw"),
                update_strategy=LocalizationType.ADAPTIVE,
                input_source=DataSource.SAMPLED,
            ),
        ],
    )


@pytest.mark.parametrize("priority", ["sampled", "design_matrix"])
def test_that_merge_with_existing_parameters_merges_correctly_with_overlapping_names_and_priority(  # ruff: ignore[line-too-long]
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
            parameter_type_update_strategies={"GEN_KW": LocalizationType.ADAPTIVE},
        )

    # state after merging with existing parameters
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

    gen_kw = [
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
    ]

    dm.parameter_priority = {
        cfg.name: dm.priority_source for cfg in dm.parameter_configurations
    }

    merged_params = dm.merge_with_existing_parameters(existing_parameters=gen_kw)

    if priority == "sampled":
        _assert_parameter_configurations(
            merged_params,
            [
                *gen_kw,
                GenKwConfig(
                    name="param1",
                    distribution=RawSettings(name="raw"),
                    input_source=DataSource.DESIGN_MATRIX,
                    update_strategy=LocalizationType.ADAPTIVE,
                ),
                GenKwConfig(
                    name="param3",
                    distribution=RawSettings(name="raw"),
                    input_source=DataSource.DESIGN_MATRIX,
                    update_strategy=LocalizationType.ADAPTIVE,
                ),
            ],
        )

    elif priority == "design_matrix":
        _assert_parameter_configurations(
            merged_params,
            [
                *dm.parameter_configurations,
                GenKwConfig(
                    name="param4",
                    distribution=RawSettings(name="raw"),
                    update_strategy=LocalizationType.ADAPTIVE,
                    input_source=DataSource.SAMPLED,
                ),
            ],
        )
