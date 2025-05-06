import os.path

import numpy as np
import pytest
from orjson import orjson
from pydantic import ValidationError

import everest.optimizer.everest2ropt as everest2ropt_patch
from everest.config import EverestConfig
from everest.config_file_loader import yaml_file_to_substituted_config_dict
from everest.optimizer.everest2ropt import _everest2ropt, everest2ropt
from everest.optimizer.opt_model_transforms import get_optimization_domain_transforms
from tests.everest.utils import relpath

_CONFIG_DIR = relpath("test_data/mocked_test_case")
_CONFIG_FILE = "mocked_test_case.yml"


def test_tutorial_everest2ropt():
    ever_config = EverestConfig.load_file(os.path.join(_CONFIG_DIR, _CONFIG_FILE))
    ropt_config = everest2ropt(ever_config)

    realizations = ropt_config.realizations

    assert len(realizations.weights) == 2
    assert realizations.weights[0] == 0.5


def test_everest2ropt_controls():
    config = EverestConfig.load_file(os.path.join(_CONFIG_DIR, _CONFIG_FILE))

    controls = config.controls
    assert len(controls) == 1

    ropt_config = everest2ropt(config)

    assert len(ropt_config.variables.lower_bounds) == 16
    assert len(ropt_config.variables.upper_bounds) == 16


def test_everest2ropt_controls_auto_scale():
    config = EverestConfig.load_file(os.path.join(_CONFIG_DIR, _CONFIG_FILE))
    controls = config.controls
    controls[0].scaled_range = [0.3, 0.7]
    ropt_config = everest2ropt(
        config,
        transforms=get_optimization_domain_transforms(
            config.controls,
            config.objective_functions,
            config.output_constraints,
            config.model,
        ),
    )
    assert np.allclose(ropt_config.variables.lower_bounds, 0.3)
    assert np.allclose(ropt_config.variables.upper_bounds, 0.7)


def test_everest2ropt_controls_input_constraint():
    config = EverestConfig.load_file(
        os.path.join(_CONFIG_DIR, "config_input_constraints.yml")
    )
    input_constraints_ever_config = config.input_constraints
    # Check that there are two input constraints entries in the config
    assert len(input_constraints_ever_config) == 2

    ropt_config = everest2ropt(config)

    # The input has two constraints: one two-sided inequality constraint,
    # and an equality constraint. The first is converted into LE and GE
    # constraints by Everest, so the ropt input should contain three
    # constraints: LE, GE and EQ.

    # Check that the config is defining three input constraints.
    assert ropt_config.linear_constraints.coefficients.shape[0] == 2

    # Check the bounds:
    exp_lower_bounds = [0.0, 1.0]
    exp_upper_bounds = [1.0, 1.0]
    assert exp_lower_bounds == ropt_config.linear_constraints.lower_bounds.tolist()
    assert exp_upper_bounds == ropt_config.linear_constraints.upper_bounds.tolist()


def test_everest2ropt_controls_input_constraint_auto_scale():
    config = EverestConfig.load_file(
        os.path.join(_CONFIG_DIR, "config_input_constraints.yml")
    )
    input_constraints_ever_config = config.input_constraints
    # Check that there are two input constraints entries in the config
    assert len(input_constraints_ever_config) == 2

    ropt_config = everest2ropt(config)
    min_values = ropt_config.variables.lower_bounds.copy()
    max_values = ropt_config.variables.upper_bounds.copy()
    coefficients = ropt_config.linear_constraints.coefficients
    lower_bounds = ropt_config.linear_constraints.lower_bounds
    upper_bounds = ropt_config.linear_constraints.upper_bounds

    controls = config.controls
    min_values[1] = -1.0
    max_values[1] = 1.0
    for idx in range(3):
        controls[0].variables[idx].min = min_values[idx]
        controls[0].variables[idx].max = max_values[idx]
    controls[0].auto_scale = True
    controls[0].scaled_range = [0.3, 0.7]

    scaled_lower_bounds = lower_bounds - np.matmul(
        coefficients, min_values - 0.3 * (max_values - min_values) / 0.4
    )
    scaled_upper_bounds = upper_bounds - np.matmul(
        coefficients, min_values - 0.3 * (max_values - min_values) / 0.4
    )
    scaled_coefficients = coefficients * (max_values - min_values) / 0.4
    scaled_coefficients[:2, 1] = coefficients[:2, 1] * 2.0 / 0.4

    ropt_config = everest2ropt(
        config,
        transforms=get_optimization_domain_transforms(
            config.controls,
            config.objective_functions,
            config.output_constraints,
            config.model,
        ),
    )
    assert np.allclose(
        ropt_config.linear_constraints.coefficients,
        scaled_coefficients,
    )
    assert np.allclose(
        ropt_config.linear_constraints.lower_bounds[0],
        scaled_lower_bounds[0],
    )
    assert np.allclose(
        ropt_config.linear_constraints.upper_bounds[0],
        scaled_upper_bounds[0],
    )


def test_everest2ropt_controls_optimizer_setting():
    config = os.path.join(_CONFIG_DIR, "config_full_gradient_info.yml")
    config = EverestConfig.load_file(config)
    ropt_config = everest2ropt(config)
    assert len(ropt_config.realizations.weights) == 15
    assert ropt_config.optimizer.method == "optpp_q_newton"
    assert ropt_config.gradient.number_of_perturbations == 20


def test_everest2ropt_constraints():
    config = os.path.join(_CONFIG_DIR, "config_output_constraints.yml")
    config = EverestConfig.load_file(config)
    ropt_config = everest2ropt(config)
    assert len(ropt_config.nonlinear_constraints.lower_bounds) == 16


def test_everest2ropt_backend_options():
    config = os.path.join(_CONFIG_DIR, "config_output_constraints.yml")
    config = EverestConfig.load_file(config)

    config.optimization.options = ["test = 1"]
    ropt_config = everest2ropt(config)
    assert ropt_config.optimizer.options == ["test = 1"]

    config.optimization.backend_options = {"test": "5"}  # should be disregarded
    ropt_config = everest2ropt(config)
    assert ropt_config.optimizer.options == ["test = 1"]

    config.optimization.options = None
    ropt_config = everest2ropt(config)
    assert ropt_config.optimizer.options == {"test": "5"}

    config.optimization.options = ["hey", "a=b", "c 100"]
    ropt_config = everest2ropt(config)
    assert ropt_config.optimizer.options == ["hey", "a=b", "c 100"]


def test_everest2ropt_samplers():
    config = os.path.join(_CONFIG_DIR, "config_samplers.yml")
    config = EverestConfig.load_file(config)

    ropt_config = everest2ropt(config)

    assert len(ropt_config.samplers) == 5
    assert ropt_config.gradient.samplers.tolist() == [0, 0, 1, 2, 3, 4]
    assert ropt_config.samplers[0].method == "norm"
    assert ropt_config.samplers[1].method == "norm"
    assert ropt_config.samplers[2].method == "uniform"
    assert ropt_config.samplers[3].method == "norm"
    assert ropt_config.samplers[4].method == "uniform"
    for idx in range(5):
        if idx == 1:
            assert ropt_config.samplers[idx].shared
        else:
            assert not ropt_config.samplers[idx].shared


def test_everest2ropt_cvar():
    config_dict = yaml_file_to_substituted_config_dict(
        os.path.join(_CONFIG_DIR, _CONFIG_FILE)
    )

    config_dict["optimization"]["cvar"] = {}

    with pytest.raises(ValidationError, match="Invalid CVaR section"):
        EverestConfig.model_validate(config_dict)

    config_dict["optimization"]["cvar"] = {
        "percentile": 0.1,
        "number_of_realizations": 1,
    }

    with pytest.raises(ValidationError, match=r".*Invalid CVaR section.*"):
        EverestConfig.model_validate(config_dict)

    config_dict["optimization"]["cvar"] = {
        "number_of_realizations": 1,
    }

    ropt_config = everest2ropt(EverestConfig.model_validate(config_dict))

    assert ropt_config.objectives.realization_filters == [0]
    assert len(ropt_config.realization_filters) == 1
    assert ropt_config.realization_filters[0].method == "sort-objective"
    assert ropt_config.realization_filters[0].options["sort"] == [0]
    assert ropt_config.realization_filters[0].options["first"] == 0
    assert ropt_config.realization_filters[0].options["last"] == 0

    config_dict["optimization"]["cvar"] = {
        "percentile": 0.3,
    }

    ropt_config = everest2ropt(EverestConfig.model_validate(config_dict))
    assert ropt_config.objectives.realization_filters == [0]
    assert len(ropt_config.realization_filters) == 1
    assert ropt_config.realization_filters[0].method == "cvar-objective"
    assert ropt_config.realization_filters[0].options["sort"] == [0]
    assert ropt_config.realization_filters[0].options["percentile"] == 0.3


def test_everest2ropt_arbitrary_backend_options():
    config = EverestConfig.load_file(os.path.join(_CONFIG_DIR, _CONFIG_FILE))
    config.optimization.backend_options = {"a": [1]}

    ropt_config = everest2ropt(config)
    assert "a" in ropt_config.optimizer.options
    assert ropt_config.optimizer.options["a"] == [1]


def test_everest2ropt_default_algorithm_name(min_config):
    config = EverestConfig(**min_config)
    assert not min_config.get("optimization")
    ropt_config = everest2ropt(config)
    assert ropt_config.optimizer.method == "optpp_q_newton"


@pytest.mark.parametrize(
    "case", ["config_advanced.yml", "config_multiobj.yml", "config_minimal.yml"]
)
def test_everest2ropt_snapshot(case, snapshot):
    config = EverestConfig.load_file(
        relpath(f"../../test-data/everest/math_func/{case}")
    )
    ropt_config = everest2ropt(
        config,
        transforms=get_optimization_domain_transforms(
            config.controls,
            config.objective_functions,
            config.output_constraints,
            config.model,
        ),
    ).model_dump()

    def safe_default(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert numpy array to a list
        elif isinstance(
            obj, np.generic
        ):  # Handles numpy scalars like np.int32, np.float64
            return obj.item()
        else:
            return str(obj)

        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    ropt_config["optimizer"]["output_dir"] = "not_relevant"

    ropt_config_str = (
        orjson.dumps(
            ropt_config,
            option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS,
            default=safe_default,
        )
        .decode("utf-8")
        .strip()
        + "\n"
    )
    snapshot.assert_match(ropt_config_str, "ropt_config.json")


def test_everest2ropt_validation_error(monkeypatch) -> None:
    def _patched_everest2ropt(ever_config):
        ropt_dict = _everest2ropt(ever_config)
        ropt_dict["foo"] = "bar"
        return ropt_dict

    ever_config = EverestConfig.load_file(os.path.join(_CONFIG_DIR, _CONFIG_FILE))
    monkeypatch.setattr(everest2ropt_patch, "_everest2ropt", _patched_everest2ropt)
    with pytest.raises(ValueError, match=r"Validation error\(s\) in ropt"):
        everest2ropt_patch.everest2ropt(ever_config)
