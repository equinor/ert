from copy import deepcopy

import pytest
import ert3


@pytest.fixture()
def base_ensemble_config():
    yield {
        "size": 1000,
        "inputs": [{"source": "stochastic.coefficients", "record": "coefficients"}],
        "forward_model": {
            "driver": "local",
            "stage": "evaluate_polynomial",
        },
    }


def test_entry_point(base_ensemble_config):
    config = ert3.config.load_ensemble_config(base_ensemble_config)
    assert config.size == 1000
    assert config.forward_model.driver == "local"
    assert config.forward_model.stage == "evaluate_polynomial"


@pytest.mark.parametrize("driver", ["local", "pbs"])
def test_config(driver, base_ensemble_config):
    config_dict = deepcopy(base_ensemble_config)
    config_dict["forward_model"]["driver"] = driver
    config = ert3.config.load_ensemble_config(config_dict)
    assert config.size == 1000
    assert config.forward_model.driver == driver


def test_forward_model_default_driver(base_ensemble_config):
    base_ensemble_config["forward_model"].pop("driver")
    config = ert3.config.load_ensemble_config(base_ensemble_config)
    assert config.forward_model.driver == "local"


def test_forward_model_invalid_driver(base_ensemble_config):
    base_ensemble_config["forward_model"] = {
        "driver": "not_installed_driver",
        "stage": "some_name",
    }

    with pytest.raises(
        ert3.exceptions.ConfigValidationError,
        match="unexpected value; permitted: 'local'",
    ):
        ert3.config.load_ensemble_config(base_ensemble_config)


@pytest.mark.parametrize(
    "input_config, expected_source, expected_record",
    [
        ({"source": "some.source", "record": "coeffs"}, "some.source", "coeffs"),
    ],
)
def test_input(input_config, expected_source, expected_record, base_ensemble_config):
    base_ensemble_config["inputs"] = [input_config]
    config = ert3.config.load_ensemble_config(base_ensemble_config)
    assert config.inputs[0].source == expected_source
    assert config.inputs[0].record == expected_record


@pytest.mark.parametrize(
    "input_config, expected_error",
    [
        ({}, "2 validation errors for EnsembleConfig"),
        ({"record": "coeffs"}, "source\n  field required"),
        ({"source": "some.source"}, "record\n  field required"),
    ],
)
def test_invalid_input(input_config, expected_error, base_ensemble_config):
    base_ensemble_config["inputs"] = [input_config]
    with pytest.raises(ert3.exceptions.ConfigValidationError, match=expected_error):
        ert3.config.load_ensemble_config(base_ensemble_config)


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
        config.inputs[0].source = "different.source"

    with pytest.raises(TypeError, match="does not support item assignment"):
        config.inputs[0] = None


def test_unknown_field_in_input(base_ensemble_config):
    base_ensemble_config["inputs"][0]["unknown"] = "field"
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
