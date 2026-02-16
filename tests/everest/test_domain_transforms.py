import numpy as np
import pytest

from everest.config import EverestConfig
from everest.optimizer.everest2ropt import everest2ropt
from everest.optimizer.opt_model_transforms import get_optimization_domain_transforms
from tests.everest.utils import everest_config_with_defaults


@pytest.fixture
def ever_config() -> EverestConfig:
    return everest_config_with_defaults(
        controls=[
            {
                "name": "default",
                "type": "generic_control",
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
    )
    transforms = get_optimization_domain_transforms(
        [ctrl for c in ever_config.controls for ctrl in c.to_ert_parameter_config()],
        ever_config.create_ert_objectives_config(),
        ever_config.input_constraints,
        ever_config.create_ert_output_constraints_config(),
        ever_config.model,
        False,
    )
    assert np.allclose(
        transforms["control_scaler"].to_optimizer(
            np.asarray(ropt_config["variables"]["lower_bounds"])
        ),
        0.3,
    )
    assert np.allclose(
        transforms["control_scaler"].to_optimizer(
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

    transforms = get_optimization_domain_transforms(
        controls,
        ever_config.create_ert_objectives_config(),
        ever_config.input_constraints,
        ever_config.create_ert_output_constraints_config(),
        ever_config.model,
        scaling == "auto-scale",
    )

    coefficients = np.asarray(ropt_config["linear_constraints"]["coefficients"])
    lower_bounds = np.asarray(ropt_config["linear_constraints"]["lower_bounds"])
    upper_bounds = np.asarray(ropt_config["linear_constraints"]["upper_bounds"])

    transformed_coefficients, transformed_lower_bounds, transformed_upper_bounds = (
        transforms["control_scaler"].linear_constraints_to_optimizer(
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


def test_objective_no_scaling(ever_config):
    transforms = get_optimization_domain_transforms(
        [ctrl for c in ever_config.controls for ctrl in c.to_ert_parameter_config()],
        ever_config.create_ert_objectives_config(),
        ever_config.input_constraints,
        ever_config.create_ert_output_constraints_config(),
        ever_config.model,
        False,
    )
    transforms["objective_scaler"].calculate_auto_scales([4.0, 1.0], [0, 1])
    assert np.all(
        transforms["objective_scaler"].to_optimizer(np.asarray([1.0, 2.0])) == [-1, -2]
    )


def test_objective_manual_scaling(ever_config):
    objectives_config = ever_config.create_ert_objectives_config()
    objectives_config.scales[0] = 2.0
    transforms = get_optimization_domain_transforms(
        [ctrl for c in ever_config.controls for ctrl in c.to_ert_parameter_config()],
        objectives_config,
        ever_config.input_constraints,
        ever_config.create_ert_output_constraints_config(),
        ever_config.model,
        False,
    )
    transforms["objective_scaler"].calculate_auto_scales(
        np.asarray([4.0, 1.0]), np.asarray(ever_config.model.realizations)
    )
    assert np.all(
        transforms["objective_scaler"].to_optimizer(np.asarray([1.0, 2.0]))
        == [-0.5, -2]
    )


def test_objective_auto_scaling(ever_config):
    transforms = get_optimization_domain_transforms(
        [ctrl for c in ever_config.controls for ctrl in c.to_ert_parameter_config()],
        ever_config.create_ert_objectives_config(),
        ever_config.input_constraints,
        ever_config.create_ert_output_constraints_config(),
        ever_config.model,
        True,
    )
    transforms["objective_scaler"].calculate_auto_scales(
        np.asarray([[5.0, 1.25], [5.0, 1.25]]), [0, 1]
    )

    assert np.all(
        transforms["objective_scaler"].to_optimizer(np.asarray([1.0, 2.0]))
        == [-0.5, -1]
    )


def test_that_objective_auto_scaling_with_zero_realization_weights_fails(ever_config):
    ever_config.model.realizations_weights = [0.0, 0.0]
    transforms = get_optimization_domain_transforms(
        [ctrl for c in ever_config.controls for ctrl in c.to_ert_parameter_config()],
        ever_config.create_ert_objectives_config(),
        ever_config.input_constraints,
        ever_config.create_ert_output_constraints_config(),
        ever_config.model,
        True,
    )
    with pytest.raises(
        RuntimeError,
        match="Auto-scaling of the objective failed to estimate a positive scale",
    ):
        transforms["objective_scaler"].calculate_auto_scales(
            np.asarray([[5.0, 1.25], [5.0, 1.25]]), [0, 1]
        )


def test_that_objective_auto_scaling_with_zero_objectives_fails(ever_config):
    transforms = get_optimization_domain_transforms(
        [ctrl for c in ever_config.controls for ctrl in c.to_ert_parameter_config()],
        ever_config.create_ert_objectives_config(),
        ever_config.input_constraints,
        ever_config.create_ert_output_constraints_config(),
        ever_config.model,
        True,
    )
    with pytest.raises(
        RuntimeError,
        match="Auto-scaling of the objective failed to estimate a positive scale",
    ):
        transforms["objective_scaler"].calculate_auto_scales(np.zeros((2, 2)), [0, 1])


def test_that_infinite_objectives_are_converted_to_nan(ever_config):
    transforms = get_optimization_domain_transforms(
        [param for c in ever_config.controls for param in c.to_ert_parameter_config()],
        ever_config.create_ert_objectives_config(),
        ever_config.input_constraints,
        ever_config.create_ert_output_constraints_config(),
        ever_config.model,
        False,
    )
    transforms["objective_scaler"].calculate_auto_scales([4.0, 1.0], [0, 1])
    assert np.all(
        np.isnan(
            transforms["objective_scaler"].to_optimizer(np.asarray([-np.inf, np.inf]))
        )
    )


def test_output_constraint_no_scaling(ever_config):
    transforms = get_optimization_domain_transforms(
        [ctrl for c in ever_config.controls for ctrl in c.to_ert_parameter_config()],
        ever_config.create_ert_objectives_config(),
        ever_config.input_constraints,
        ever_config.create_ert_output_constraints_config(),
        ever_config.model,
        False,
    )
    transforms["constraint_scaler"].calculate_auto_scales([2.0, 1.0], [0, 1])
    assert np.all(
        transforms["constraint_scaler"].to_optimizer(np.asarray([1.0, 2.0])) == [1, 2]
    )


def test_output_constraint_manual_scaling(ever_config):
    constraints_config = ever_config.create_ert_output_constraints_config()
    constraints_config.scales[0] = 2.0
    transforms = get_optimization_domain_transforms(
        [ctrl for c in ever_config.controls for ctrl in c.to_ert_parameter_config()],
        ever_config.create_ert_objectives_config(),
        ever_config.input_constraints,
        constraints_config,
        ever_config.model,
        False,
    )
    transforms["constraint_scaler"].calculate_auto_scales(
        np.asarray([4.0, 1.0]), np.asarray(ever_config.model.realizations)
    )
    assert np.all(
        transforms["constraint_scaler"].to_optimizer(np.asarray([1.0, 2.0])) == [0.5, 2]
    )


def test_output_constraint_auto_scaling(ever_config):
    transforms = get_optimization_domain_transforms(
        [ctrl for c in ever_config.controls for ctrl in c.to_ert_parameter_config()],
        ever_config.create_ert_objectives_config(),
        ever_config.input_constraints,
        ever_config.create_ert_output_constraints_config(),
        ever_config.model,
        True,
    )
    transforms["constraint_scaler"].calculate_auto_scales(
        np.asarray([[4.0, 2.0], [4.0, 2.0]]), [0, 1]
    )

    assert np.all(
        transforms["constraint_scaler"].to_optimizer(np.asarray([1.0, 2.0]))
        == [0.25, 1.0]
    )


def test_that_output_constraint_auto_scaling_with_zero_realization_weights_fails(
    ever_config,
):
    ever_config.model.realizations_weights = [0.0, 0.0]
    transforms = get_optimization_domain_transforms(
        [ctrl for c in ever_config.controls for ctrl in c.to_ert_parameter_config()],
        ever_config.create_ert_objectives_config(),
        ever_config.input_constraints,
        ever_config.create_ert_output_constraints_config(),
        ever_config.model,
        True,
    )
    with pytest.raises(
        RuntimeError,
        match="Auto-scaling of the constraints failed to estimate a positive scale",
    ):
        transforms["constraint_scaler"].calculate_auto_scales(
            np.asarray([[4.0, 2.0], [4.0, 2.0]]), [0, 1]
        )


def test_that_output_constraint_auto_scaling_with_zero_constraints_fails(ever_config):
    transforms = get_optimization_domain_transforms(
        [ctrl for c in ever_config.controls for ctrl in c.to_ert_parameter_config()],
        ever_config.create_ert_objectives_config(),
        ever_config.input_constraints,
        ever_config.create_ert_output_constraints_config(),
        ever_config.model,
        True,
    )
    with pytest.raises(
        RuntimeError,
        match="Auto-scaling of the constraints failed to estimate a positive scale",
    ):
        transforms["constraint_scaler"].calculate_auto_scales(np.zeros((2, 2)), [0, 1])


def test_that_infinite_output_constraints_are_converted_to_nan(ever_config):
    transforms = get_optimization_domain_transforms(
        [param for c in ever_config.controls for param in c.to_ert_parameter_config()],
        ever_config.create_ert_objectives_config(),
        ever_config.input_constraints,
        ever_config.create_ert_output_constraints_config(),
        ever_config.model,
        False,
    )
    transforms["constraint_scaler"].calculate_auto_scales([2.0, 1.0], [0, 1])
    assert np.all(
        np.isnan(
            transforms["constraint_scaler"].to_optimizer(np.asarray([-np.inf, np.inf]))
        )
    )
