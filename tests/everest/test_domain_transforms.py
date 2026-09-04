import numpy as np
import pytest

from everest.config import EverestConfig
from everest.optimizer.everest2ropt import everest2ropt
from everest.optimizer.opt_model_transforms import get_control_scaler
from tests.everest.utils import everest_config_with_defaults


@pytest.fixture
def ever_config() -> EverestConfig:
    return everest_config_with_defaults(
        controls=[
            {
                "name": "default",
                "min": 0,
                "max": 1.0,
                "scaled_range": [0.3, 0.7],
                "initial_guess": 0.5,
                "perturbation_magnitude": 0.01,
                "variables": [
                    {"name": "a"},
                    {"name": "b"},
                    {"name": "c"},
                    {"name": "e"},
                    {"name": "f"},
                    {"name": "g"},
                ],
            }
        ],
        objective_functions=[
            {"name": "f1", "weight": 1.0},
            {"name": "f2", "weight": 4.0},
        ],
        input_constraints=[
            {
                "upper_bound": 1,
                "lower_bound": 0,
                "weights": {"default.a": 0.1, "default.b": 0.2, "default.c": 0.3},
            },
            {
                "target": 1,
                "weights": {"default.e": 1.0, "default.f": 1.0, "default.g": 1.0},
            },
        ],
        output_constraints=[
            {"name": "c1", "upper_bound": 1.0},
            {"name": "c2", "upper_bound": 1.0},
        ],
        model={"realizations": [0, 1], "realizations_weights": [0.5, 0.5]},
    )


def test_transforms_controls_scaling(ever_config):
    ropt_config, _ = everest2ropt(
        [ctrl for c in ever_config.controls for ctrl in c.to_ert_parameter_config()],
        ever_config.create_ert_objectives_config(),
        ever_config.input_constraints,
        ever_config.create_ert_output_constraints_config(),
        ever_config.optimization,
        ever_config.model,
        ever_config.environment.random_seed,
        ever_config.optimization_output_dir,
        None,
    )
    control_scaler = get_control_scaler(
        [ctrl for c in ever_config.controls for ctrl in c.to_ert_parameter_config()],
        ever_config.input_constraints,
        False,
    )
    assert np.allclose(
        control_scaler.to_optimizer(
            np.asarray(ropt_config["variables"]["lower_bounds"])
        ),
        0.3,
    )
    assert np.allclose(
        control_scaler.to_optimizer(
            np.asarray(ropt_config["variables"]["upper_bounds"])
        ),
        0.7,
    )


@pytest.mark.parametrize("scaling", ["none", "manual", "auto-scale"])
def test_transforms_controls_input_constraint_scaling(ever_config, scaling):
    input_constraints_ever_config = ever_config.input_constraints
    assert len(input_constraints_ever_config) == 2

    ever_config.optimization.auto_scale = scaling == "auto-scale"
    if scaling == "manual":
        input_constraints_ever_config[1].scale = 2.0

    ropt_config, _ = everest2ropt(
        [ctrl for c in ever_config.controls for ctrl in c.to_ert_parameter_config()],
        ever_config.create_ert_objectives_config(),
        ever_config.input_constraints,
        ever_config.create_ert_output_constraints_config(),
        ever_config.optimization,
        ever_config.model,
        ever_config.environment.random_seed,
        ever_config.optimization_output_dir,
        None,
    )

    controls = [
        ctrl for c in ever_config.controls for ctrl in c.to_ert_parameter_config()
    ]
    min_values = np.asarray(ropt_config["variables"]["lower_bounds"])
    max_values = np.asarray(ropt_config["variables"]["upper_bounds"])
    min_values[1] = -1.0
    max_values[1] = 1.0
    for idx in range(3):
        controls[idx].min = min_values[idx]
        controls[idx].max = max_values[idx]

    for control in controls:
        control.scaled_range = [0.3, 0.7]

    transforms = get_control_scaler(
        controls,
        ever_config.input_constraints,
        scaling == "auto-scale",
    )

    coefficients = np.asarray(ropt_config["linear_constraints"]["coefficients"])
    lower_bounds = np.asarray(ropt_config["linear_constraints"]["lower_bounds"])
    upper_bounds = np.asarray(ropt_config["linear_constraints"]["upper_bounds"])

    transformed_coefficients, transformed_lower_bounds, transformed_upper_bounds = (
        transforms.linear_constraints_to_optimizer(
            coefficients, lower_bounds, upper_bounds
        )
    )

    scaled_lower_bounds = lower_bounds - np.matmul(
        coefficients, min_values - 0.3 * (max_values - min_values) / 0.4
    )
    scaled_upper_bounds = upper_bounds - np.matmul(
        coefficients, min_values - 0.3 * (max_values - min_values) / 0.4
    )
    scaled_coefficients = coefficients * (max_values - min_values) / 0.4
    scaled_coefficients[:, 1] = coefficients[:, 1] * 2.0 / 0.4

    match scaling:
        case "none":
            scales = np.array([1.0, 1.0])
        case "manual":
            scales = np.array([1.0, 2.0])
        case "auto-scale":
            b_max = np.maximum(np.abs(scaled_lower_bounds), np.abs(scaled_upper_bounds))
            c_max = np.max(np.abs(scaled_coefficients), axis=1)
            scales = np.maximum(b_max, c_max)
    scaled_lower_bounds /= scales
    scaled_upper_bounds /= scales
    scaled_coefficients /= scales[:, np.newaxis]

    assert np.allclose(transformed_lower_bounds, scaled_lower_bounds)
    assert np.allclose(transformed_upper_bounds, scaled_upper_bounds)
    assert np.allclose(transformed_coefficients, scaled_coefficients)


def _ropt_config(ever_config, objectives=None, output_constraints=None):
    ropt_config, _ = everest2ropt(
        [ctrl for c in ever_config.controls for ctrl in c.to_ert_parameter_config()],
        ever_config.create_ert_objectives_config()
        if objectives is None
        else objectives,
        ever_config.input_constraints,
        ever_config.create_ert_output_constraints_config()
        if output_constraints is None
        else output_constraints,
        ever_config.optimization,
        ever_config.model,
        ever_config.environment.random_seed,
        ever_config.optimization_output_dir,
        None,
    )
    return ropt_config


def test_that_objective_scales_are_passed_to_ropt(ever_config):
    objectives_config = ever_config.create_ert_objectives_config()
    objectives_config.scales[0] = 2.0
    ropt_config = _ropt_config(ever_config, objectives=objectives_config)
    assert ropt_config["objectives"]["scales"] == [2.0, 1.0]
    assert not ropt_config["objectives"]["auto_scale"]


def test_that_mean_objectives_are_maximized(ever_config):
    ropt_config = _ropt_config(ever_config)
    assert ropt_config["objectives"]["maximize"] == [True, True]


def test_that_a_spread_objective_is_minimized(ever_config):
    objectives_config = ever_config.create_ert_objectives_config()
    objectives_config.objective_types[1] = "stddev"
    ropt_config = _ropt_config(ever_config, objectives=objectives_config)
    # The direction is applied to the aggregate, so maximizing a spread would
    # ask for the least robust solution rather than the most robust one.
    assert ropt_config["objectives"]["maximize"] == [True, False]


def test_that_auto_scale_enables_it_for_objectives_and_output_constraints(ever_config):
    ever_config.optimization.auto_scale = True
    ropt_config = _ropt_config(ever_config)
    assert ropt_config["objectives"]["auto_scale"]
    assert ropt_config["nonlinear_constraints"]["auto_scale"]


def test_that_output_constraint_scales_are_passed_to_ropt(ever_config):
    constraints_config = ever_config.create_ert_output_constraints_config()
    constraints_config.scales[0] = 2.0
    ropt_config = _ropt_config(ever_config, output_constraints=constraints_config)
    assert ropt_config["nonlinear_constraints"]["scales"] == [2.0, 1.0]
    assert not ropt_config["nonlinear_constraints"]["auto_scale"]
