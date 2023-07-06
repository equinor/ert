import pytest
from pydantic import ValidationError

from ert._c_wrappers.analysis.configuration import UpdateConfiguration
from ert._c_wrappers.enkf.row_scaling import RowScaling


def test_configuration():
    config = UpdateConfiguration(
        update_steps=[
            {
                "name": "update_step_name",
                "observations": ["MY_OBS"],
                "parameters": ["MY_PARAMETER"],
                "row_scaling_parameters": [("MY_ROW_SCALING", RowScaling())],
            }
        ]
    )

    config.context_validate(["MY_OBS"], ["MY_PARAMETER", "MY_ROW_SCALING"])


@pytest.mark.parametrize(
    "config, expectation",
    [
        [
            {"name": "not_relevant", "observations": ["not_relevant"]},
            pytest.raises(ValidationError, match="Must provide at least one parameter"),
        ],
        [
            {"name": "not_relevant", "parameters": ["not_relevant"]},
            pytest.raises(ValidationError, match="update_steps -> 0 -> observations"),
        ],
        [
            {
                "name": "not_relevant",
                "observations": ["not_relevant"],
                "parameters": "relevant",
            },
            pytest.raises(ValidationError, match="value is not a valid list"),
        ],
        [
            {
                "name": "not_relevant",
                "observations": "relevant",
                "parameters": ["not_relevant"],
            },
            pytest.raises(ValidationError, match="value is not a valid list"),
        ],
        [
            {
                "name": "not_relevant",
                "observations": ["relevant"],
                "parameters": [],
            },
            pytest.raises(ValidationError, match="Must provide at least one parameter"),
        ],
        [
            {
                "name": "not_relevant",
                "observations": [],
                "parameters": ["not_relevant"],
            },
            pytest.raises(
                ValidationError, match=" ensure this value has at least 1 item"
            ),
        ],
        [
            {
                "observations": ["not_relevant"],
                "parameters": ["not_relevant"],
            },
            pytest.raises(ValidationError, match="update_steps -> 0 -> name"),
        ],
    ],
)
def test_missing(config, expectation):
    with expectation:
        UpdateConfiguration(update_steps=[config])


@pytest.mark.parametrize(
    "config, expectation",
    [
        [
            [
                {
                    "name": "not_relevant",
                    "observations": ["not_relevant"],
                    "parameters": "not_list",
                },
                {"name": "not_relevant", "parameters": ["not_relevant"]},
            ],
            pytest.raises(
                ValidationError, match="2 validation errors for UpdateConfiguration"
            ),
        ],
        [
            [
                {},
                {"name": "not_relevant", "observations": ["not_relevant"]},
            ],
            pytest.raises(
                ValidationError, match="3 validation errors for UpdateConfiguration"
            ),
        ],
    ],
)
def test_missing_multiple(config, expectation):
    with expectation:
        UpdateConfiguration(update_steps=config)


@pytest.mark.parametrize(
    "input_obs",
    [
        ["OBS"],
        [{"name": "OBS"}],
        [{"name": "OBS", "index_list": [1, 2, 3]}],
        [["OBS", [1, 2, 3]]],
        [("OBS", [1, 2, 3])],
        ["OBS_1", ["OBS", [1, 2, 3]]],
    ],
)
def test_configuration_valid_obs_input(input_obs):
    config = UpdateConfiguration(
        update_steps=[
            {
                "name": "not_relevant",
                "observations": input_obs,
                "parameters": ["not_relevant"],
            }
        ]
    )
    config.context_validate(["OBS", "OBS_1"], ["not_relevant"])


def test_user_setup():
    test_input = [
        {
            "name": "update_step_NAME",
            "observations": [
                "WOPR_OP1_72",
                ("MY_INDEX_OBS", [1, 2, 3]),
                ("JUST OBS"),
            ],
            "parameters": ["SNAKE_OIL_PARAM", ("INDEX_PARAMETER", [1, 2, 3])],
            "row_scaling_parameters": [
                ("ROW_SCALE", RowScaling()),
                ("ROW_SCALE", RowScaling(), [1, 2, 3]),
            ],
        }
    ]
    UpdateConfiguration(update_steps=test_input)
