import pytest
import pydantic
from ert3.config import _ensemble_config
from copy import deepcopy

_config_dict = {
    "size": 1000,
    "input": [{"source": "stochastic.coefficients", "record": "coefficients"}],
    "forward_model": {
        "driver": "local",
        "stages": [
            "evaluate_polynomial",
        ],
    },
}


def test_entry_point():
    config = _ensemble_config.load_ensemble_config(_config_dict)
    assert config.size == 1000
    assert config.forward_model.driver == "local"
    assert config.forward_model.stages == ["evaluate_polynomial"]


@pytest.mark.parametrize("driver", ["local", "pbs"])
def test_config(driver):
    config_dict = deepcopy(_config_dict)
    config_dict["forward_model"]["driver"] = driver
    config = _ensemble_config.EnsembleConfig(**config_dict)
    assert config.size == 1000
    assert config.forward_model.driver == driver


def test_forward_model_default_driver():
    config = _ensemble_config.ForwardModel(
        **{
            "stages": [
                "some_name",
            ]
        }
    )
    assert config.driver == "local"


def test_forward_model_invalid_driver():
    config = {
        "driver": "not_installed_driver",
        "stages": [
            "some_name",
        ],
    }

    with pytest.raises(
        pydantic.error_wrappers.ValidationError,
        match="unexpected value; permitted: 'local'",
    ):
        _ensemble_config.ForwardModel(**config)


@pytest.mark.parametrize(
    "input_config, expected_source, expected_record",
    [
        ({"source": "some.source", "record": "coeffs"}, "some.source", "coeffs"),
    ],
)
def test_input(input_config, expected_source, expected_record):
    config = _ensemble_config.Input(**input_config)
    assert config.source == expected_source
    assert config.record == expected_record


@pytest.mark.parametrize(
    "input_config, expected_error",
    [
        ({}, "2 validation errors for Input"),
        ({"record": "coeffs"}, "error for Input\nsource"),
        ({"source": "some.source"}, "error for Input\nrecord"),
    ],
)
def test_invalid_input(input_config, expected_error):
    with pytest.raises(pydantic.error_wrappers.ValidationError, match=expected_error):
        _ensemble_config.Input(**input_config)
