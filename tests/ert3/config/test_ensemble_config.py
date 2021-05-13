import pytest
import pydantic
import ert3
from ert3.config import _ensemble_config
from copy import deepcopy


@pytest.fixture()
def base_ensemble_config():
    yield {
        "size": 1000,
        "input": [{"source": "stochastic.coefficients", "record": "coefficients"}],
        "forward_model": {
            "driver": "local",
            "stage": "evaluate_polynomial",
        },
    }


def test_entry_point(base_ensemble_config):
    config = _ensemble_config.load_ensemble_config(base_ensemble_config)
    assert config.size == 1000
    assert config.forward_model.driver == "local"
    assert config.forward_model.stage == "evaluate_polynomial"


@pytest.mark.parametrize("driver", ["local", "pbs"])
def test_config(driver, base_ensemble_config):
    config_dict = deepcopy(base_ensemble_config)
    config_dict["forward_model"]["driver"] = driver
    config = _ensemble_config.EnsembleConfig(**config_dict)
    assert config.size == 1000
    assert config.forward_model.driver == driver


def test_forward_model_default_driver():
    config = _ensemble_config.ForwardModel(
        **{
            "stage": "some_name",
        }
    )
    assert config.driver == "local"


def test_forward_model_invalid_driver():
    config = {
        "driver": "not_installed_driver",
        "stage": "some_name",
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


def test_immutable_base(base_ensemble_config):
    config = ert3.config.load_ensemble_config(base_ensemble_config)
    with pytest.raises(TypeError, match="does not support item assignment"):
        config.size = 42


def test_unknown_field_in_base(base_ensemble_config):
    base_ensemble_config["unknown"] = "field"
    with pytest.raises(
        ert3.exceptions.ConfigValidationError, match="extra fields not permitted"
    ):
        ert3.config.load_ensemble_config(base_ensemble_config)


def test_immutable_input(base_ensemble_config):
    config = ert3.config.load_ensemble_config(base_ensemble_config)
    with pytest.raises(TypeError, match="does not support item assignment"):
        config.input[0].source = "different.source"

    with pytest.raises(TypeError, match="does not support item assignment"):
        config.input[0] = None


def test_unknown_field_in_input(base_ensemble_config):
    base_ensemble_config["input"][0]["unknown"] = "field"
    with pytest.raises(
        ert3.exceptions.ConfigValidationError, match="extra fields not permitted"
    ):
        ert3.config.load_ensemble_config(base_ensemble_config)


def test_immutable_forward_model(base_ensemble_config):
    config = ert3.config.load_ensemble_config(base_ensemble_config)
    with pytest.raises(TypeError, match="does not support item assignment"):
        config.forward_model.stage = "my_stage"


def test_unknown_field_in_forward_model(base_ensemble_config):
    base_ensemble_config["forward_model"]["unknown"] = "field"
    with pytest.raises(
        ert3.exceptions.ConfigValidationError, match="extra fields not permitted"
    ):
        ert3.config.load_ensemble_config(base_ensemble_config)
