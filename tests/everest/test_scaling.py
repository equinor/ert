import numpy as np
import pytest

from everest.config import EverestConfig
from everest.optimizer.everest2ropt import everest2ropt
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


def _controls(ever_config):
    return [ctrl for c in ever_config.controls for ctrl in c.to_ert_parameter_config()]


def _everest2ropt(ever_config):
    ropt_config, _ = everest2ropt(
        _controls(ever_config),
        ever_config.create_ert_objectives_config(),
        ever_config.input_constraints,
        ever_config.create_ert_output_constraints_config(),
        ever_config.optimization,
        ever_config.model,
        ever_config.environment.random_seed,
        ever_config.optimization_output_dir,
    )
    return ropt_config


def test_that_control_scales_map_the_bounds_onto_the_scaled_range(ever_config):
    ropt_config = _everest2ropt(ever_config)
    variables = ropt_config["variables"]
    scales = np.asarray(variables["scales"])
    offsets = np.asarray(variables["offsets"])

    # ropt applies (x - offset) / scale, so the bounds land on scaled_range.
    lower_bounds = np.asarray(variables["lower_bounds"])
    upper_bounds = np.asarray(variables["upper_bounds"])
    assert np.allclose((lower_bounds - offsets) / scales, 0.3)
    assert np.allclose((upper_bounds - offsets) / scales, 0.7)


def test_that_an_integer_control_is_not_scaled(ever_config):
    ever_config.controls[0].control_type = "integer"
    ropt_config = _everest2ropt(ever_config)
    assert np.allclose(ropt_config["variables"]["scales"], 1.0)
    assert np.allclose(ropt_config["variables"]["offsets"], 0.0)


def test_that_a_disabled_control_is_scaled_like_any_other(ever_config):
    ever_config.controls[0].variables[1].enabled = False
    ropt_config = _everest2ropt(ever_config)
    scales = np.asarray(ropt_config["variables"]["scales"])
    offsets = np.asarray(ropt_config["variables"]["offsets"])
    assert np.allclose(scales, scales[0])
    assert np.allclose(offsets, offsets[0])


def test_that_input_constraint_scales_default_to_one(ever_config):
    ropt_config = _everest2ropt(ever_config)
    assert ropt_config["linear_constraints"]["scales"] == [1.0, 1.0]
    assert not ropt_config["linear_constraints"]["auto_scale"]


def test_that_input_constraint_scales_are_passed_to_ropt(ever_config):
    ever_config.input_constraints[1].scale = 2.0
    ropt_config = _everest2ropt(ever_config)
    assert ropt_config["linear_constraints"]["scales"] == [1.0, 2.0]


def test_that_auto_scale_enables_it_for_input_constraints(ever_config):
    ever_config.optimization.auto_scale = True
    ropt_config = _everest2ropt(ever_config)
    assert ropt_config["linear_constraints"]["auto_scale"]


def test_that_auto_scale_replaces_a_configured_input_constraint_scale(ever_config):
    ever_config.input_constraints[1].scale = 2.0
    ever_config.optimization.auto_scale = True
    ropt_config = _everest2ropt(ever_config)
    assert ropt_config["linear_constraints"]["scales"] == [1.0, 1.0]


def _ropt_config(ever_config, objectives=None, output_constraints=None):
    ropt_config, _ = everest2ropt(
        _controls(ever_config),
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


def test_that_auto_scale_replaces_a_configured_objective_scale(ever_config):
    objectives_config = ever_config.create_ert_objectives_config()
    objectives_config.scales[0] = 2.0
    ever_config.optimization.auto_scale = True
    ropt_config = _ropt_config(ever_config, objectives=objectives_config)
    assert ropt_config["objectives"]["scales"] == [1.0, 1.0]


def test_that_auto_scale_replaces_a_configured_output_constraint_scale(ever_config):
    constraints_config = ever_config.create_ert_output_constraints_config()
    constraints_config.scales[0] = 2.0
    ever_config.optimization.auto_scale = True
    ropt_config = _ropt_config(ever_config, output_constraints=constraints_config)
    assert ropt_config["nonlinear_constraints"]["scales"] == [1.0, 1.0]
