import re
from copy import deepcopy

import ert
import ert3
import pytest


@pytest.fixture()
def base_ensemble_config():
    yield {
        "size": 1000,
        "active_range": "0-999",
        "input": [{"source": "stochastic.coefficients", "name": "coefficients"}],
        "output": [{"name": "polynomial_output"}],
        "forward_model": {
            "driver": "local",
            "stage": "evaluate_polynomial",
        },
    }


def test_entry_point(base_ensemble_config, plugin_registry):
    config = ert3.config.load_ensemble_config(
        base_ensemble_config, plugin_registry=plugin_registry
    )
    assert config.size == 1000
    assert config.forward_model.driver == "local"
    assert config.forward_model.stage == "evaluate_polynomial"


def test_no_plugged_configuration(base_ensemble_config):
    with pytest.raises(
        RuntimeError,
        match="this configuration must be obtained from 'create_ensemble_config'.",
    ):
        ert3.config.EnsembleConfig.parse_obj(base_ensemble_config)


@pytest.mark.parametrize("driver", ["local", "pbs"])
def test_config(driver, base_ensemble_config, plugin_registry):
    config_dict = deepcopy(base_ensemble_config)
    config_dict["forward_model"]["driver"] = driver
    config = ert3.config.load_ensemble_config(
        config_dict, plugin_registry=plugin_registry
    )
    assert config.size == 1000
    assert config.forward_model.driver == driver


def test_forward_model_default_driver(base_ensemble_config, plugin_registry):
    base_ensemble_config["forward_model"].pop("driver")
    config = ert3.config.load_ensemble_config(
        base_ensemble_config, plugin_registry=plugin_registry
    )
    assert config.forward_model.driver == "local"


def test_forward_model_invalid_driver(base_ensemble_config, plugin_registry):
    base_ensemble_config["forward_model"] = {
        "driver": "not_installed_driver",
        "stage": "some_name",
    }

    with pytest.raises(
        ert.exceptions.ConfigValidationError,
        match="unexpected value; permitted: 'local'",
    ):
        ert3.config.load_ensemble_config(
            base_ensemble_config, plugin_registry=plugin_registry
        )


@pytest.mark.parametrize(
    "size, active_range",
    [
        (1, "0"),
        (1, ""),
        (1, None),
        (1, "0-0"),
        (2, "0-1"),
        (2, "1-1"),
        (2, "1"),
        (2, "0"),
        (None, None),
        (None, "0-9999999"),
    ],
)
def test_valid_active_range(base_ensemble_config, plugin_registry, size, active_range):
    base_ensemble_config["size"] = size
    base_ensemble_config["active_range"] = active_range
    ert3.config.load_ensemble_config(
        base_ensemble_config, plugin_registry=plugin_registry
    )


@pytest.mark.parametrize(
    "size, active_range",
    [(1, "1"), (10, "-0"), (10, "0-"), (10, "0-10"), (10, "0.10"), (10, "0--10")],
)
def test_invalid_active_range(
    base_ensemble_config, plugin_registry, size, active_range
):
    base_ensemble_config["size"] = size
    base_ensemble_config["active_range"] = active_range
    with pytest.raises(ert.exceptions.ConfigValidationError):
        ert3.config.load_ensemble_config(
            base_ensemble_config, plugin_registry=plugin_registry
        )


@pytest.mark.parametrize(
    ",".join(
        (
            "input_config",
            "expected_source",
            "expected_name",
            "expected_namespace",
            "expected_location",
            "expected_transformation_cls",
        )
    ),
    [
        pytest.param(
            {"source": "some.source", "name": "coeffs"},
            "some.source",
            "coeffs",
            "",
            "",
            None,
            marks=pytest.mark.xfail(),
        ),
        (
            {"source": "stochastic.source", "name": "coeffs"},
            "stochastic.source",
            "coeffs",
            "stochastic",
            "source",
            None,
        ),
        pytest.param(
            {
                "source": "resources.some.json",
                "name": "coeffs",
            },
            "resources.some.json",
            "coeffs",
            "resources",
            "some.json",
            ert.data.CopyTransformation,
            id="copy_as_default_transformation",
        ),
        pytest.param(
            {
                "source": "resources.some.json",
                "name": "coeffs",
                "transformation": {
                    "type": "copy",
                },
            },
            "resources.some.json",
            "coeffs",
            "resources",
            "some.json",
            ert.data.CopyTransformation,
            id="explicitly_defined_copy_transformation",
        ),
        pytest.param(
            {
                "source": "resources.some.json",
                "name": "coeffs",
                "transformation": {
                    "type": "copy",
                    "location": "foo.json",
                },
            },
            "resources.some.json",
            "coeffs",
            "resources",
            "foo.json",
            ert.data.CopyTransformation,
            marks=pytest.mark.raises(
                exception=ert.exceptions.ConfigValidationError,
                match=".+Either define only the source, or let them be equal",
                match_flags=(re.MULTILINE | re.DOTALL),
            ),
            id="two_conflicting_locations",
        ),
        pytest.param(
            {
                "source": "resources.some.json",
                "name": "coeffs",
                "transformation": {
                    "type": "copy",
                    "location": "some.json",
                },
            },
            "resources.some.json",
            "coeffs",
            "resources",
            "some.json",
            ert.data.CopyTransformation,
            id="locations_are_equal",
        ),
        (
            {
                "source": "resources.some.json",
                "name": "coeffs",
                "transformation": {
                    "type": "serialization",
                },
            },
            "resources.some.json",
            "coeffs",
            "resources",
            "some.json",
            ert.data.SerializationTransformation,
        ),
        (
            {
                "source": "resources.my_folder",
                "name": "my_folder",
                "transformation": {
                    "type": "directory",
                },
            },
            "resources.my_folder",
            "my_folder",
            "resources",
            "my_folder",
            ert.data.TarTransformation,
        ),
    ],
)
def test_input(
    input_config,
    expected_source,
    expected_name,
    expected_namespace,
    expected_location,
    expected_transformation_cls,
    base_ensemble_config,
    plugin_registry,
):
    base_ensemble_config["input"] = [input_config]
    config = ert3.config.load_ensemble_config(
        base_ensemble_config, plugin_registry=plugin_registry
    )
    assert config.input[0].source == expected_source
    assert config.input[0].name == expected_name
    assert config.input[0].source_namespace == expected_namespace
    assert config.input[0].source_location == expected_location

    transformation = config.input[0].get_transformation_instance()
    if transformation:
        assert isinstance(transformation, expected_transformation_cls)
    else:
        assert (
            not expected_transformation_cls
        ), f"expected {expected_transformation_cls} got None"


@pytest.mark.parametrize(
    "input_config, expected_error",
    [
        ({}, "2 validation errors for PluggedEnsembleConfig"),
        ({"name": "coeffs"}, "source\n  field required"),
        ({"source": "storage.source"}, "name\n  field required"),
    ],
)
def test_invalid_input(
    input_config, expected_error, base_ensemble_config, plugin_registry
):
    base_ensemble_config["input"] = [input_config]
    with pytest.raises(ert.exceptions.ConfigValidationError, match=expected_error):
        ert3.config.load_ensemble_config(
            base_ensemble_config, plugin_registry=plugin_registry
        )


def test_immutable_base(base_ensemble_config, plugin_registry):
    config = ert3.config.load_ensemble_config(
        base_ensemble_config, plugin_registry=plugin_registry
    )
    with pytest.raises(TypeError, match="does not support item assignment"):
        config.size = 42


def test_unknown_field_in_base(base_ensemble_config, plugin_registry):
    base_ensemble_config["unknown"] = "field"
    with pytest.raises(
        ert.exceptions.ConfigValidationError, match="extra fields not permitted"
    ):
        ert3.config.load_ensemble_config(
            base_ensemble_config, plugin_registry=plugin_registry
        )


def test_immutable_input(base_ensemble_config, plugin_registry):
    config = ert3.config.load_ensemble_config(
        base_ensemble_config, plugin_registry=plugin_registry
    )
    with pytest.raises(TypeError, match="does not support item assignment"):
        config.input[0].source = "different.source"

    with pytest.raises(TypeError, match="does not support item assignment"):
        config.input[0] = None


def test_unknown_field_in_input(base_ensemble_config, plugin_registry):
    base_ensemble_config["input"][0]["unknown"] = "field"
    with pytest.raises(
        ert.exceptions.ConfigValidationError, match="extra fields not permitted"
    ):
        ert3.config.load_ensemble_config(
            base_ensemble_config, plugin_registry=plugin_registry
        )


def test_immutable_forward_model(base_ensemble_config, plugin_registry):
    config = ert3.config.load_ensemble_config(
        base_ensemble_config, plugin_registry=plugin_registry
    )
    with pytest.raises(TypeError, match="does not support item assignment"):
        config.forward_model.stage = "my_stage"


def test_unknown_field_in_forward_model(base_ensemble_config, plugin_registry):
    base_ensemble_config["forward_model"]["unknown"] = "field"
    with pytest.raises(
        ert.exceptions.ConfigValidationError, match="extra fields not permitted"
    ):
        ert3.config.load_ensemble_config(
            base_ensemble_config, plugin_registry=plugin_registry
        )


def test_missing_ouput(base_ensemble_config, plugin_registry):
    remove_output = base_ensemble_config.copy()
    remove_output.pop("output")
    with pytest.raises(
        ert.exceptions.ConfigValidationError, match="output\n  field required"
    ):
        ert3.config.load_ensemble_config(remove_output, plugin_registry=plugin_registry)


@pytest.mark.parametrize(
    "output_config, expected_name",
    [
        ({"name": "coeffs"}, "coeffs"),
    ],
)
def test_output(output_config, expected_name, base_ensemble_config, plugin_registry):
    base_ensemble_config["output"] = [output_config]
    config = ert3.config.load_ensemble_config(
        base_ensemble_config, plugin_registry=plugin_registry
    )
    assert config.output[0].name == expected_name


@pytest.mark.parametrize(
    "output_config, expected_error",
    [
        ({}, "1 validation error for PluggedEnsembleConfig"),
        ({"something": "coeffs"}, "name\n  field required"),
    ],
)
def test_invalid_output(
    output_config, expected_error, base_ensemble_config, plugin_registry
):
    base_ensemble_config["output"] = [output_config]
    with pytest.raises(ert.exceptions.ConfigValidationError, match=expected_error):
        ert3.config.load_ensemble_config(
            base_ensemble_config, plugin_registry=plugin_registry
        )
