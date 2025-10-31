import pytest
from orjson import orjson
from pydantic import ValidationError

from ert.ensemble_evaluator.config import EvaluatorServerConfig
from ert.plugins import get_site_plugins
from ert.run_models import everest_run_model
from ert.run_models.everest_run_model import EverestRunModel
from everest.config import EverestConfig
from everest.optimizer.everest2ropt import everest2ropt
from tests.everest.conftest import everest_config_with_defaults
from tests.everest.utils import relpath


@pytest.fixture
def ever_config() -> EverestConfig:
    return everest_config_with_defaults(
        controls=[
            {
                "name": "default",
                "type": "generic_control",
                "min": 0,
                "max": 0.1,
                "perturbation_magnitude": 0.01,
                "variables": [
                    {"name": "a", "initial_guess": 0.01},
                    {
                        "name": "b",
                        "initial_guess": 0.02,
                        "sampler": {"method": "uniform"},
                    },
                    {
                        "name": "c",
                        "initial_guess": 0.03,
                        "sampler": {"method": "norm", "shared": True},
                    },
                    {"name": "e", "initial_guess": 0.04},
                    {"name": "f", "initial_guess": 0.05},
                    {"name": "g", "initial_guess": 0.06},
                ],
                "sampler": {"method": "norm"},
            }
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
            {"name": "a", "upper_bound": 1, "scale": 3},
            {"name": "b", "upper_bound": 2, "scale": 4},
        ],
        model={"realizations": [0, 1], "realizations_weights": [0.2, 0.3]},
        optimization={
            "algorithm": "optpp_q_newton",
            "perturbation_num": 2,
        },
    )


def test_tutorial_everest2ropt(ever_config):
    ropt_config, _ = everest2ropt(
        ever_config.controls,
        ever_config.objective_functions,
        ever_config.input_constraints,
        ever_config.output_constraints,
        ever_config.optimization,
        ever_config.model,
        ever_config.environment.random_seed,
        ever_config.optimization_output_dir,
    )
    realizations = ropt_config["realizations"]
    assert len(realizations["weights"]) == 2
    assert realizations["weights"][0] == 0.2


def test_everest2ropt_controls(ever_config):
    controls = ever_config.controls
    assert len(controls) == 1
    ropt_config, _ = everest2ropt(
        ever_config.controls,
        ever_config.objective_functions,
        ever_config.input_constraints,
        ever_config.output_constraints,
        ever_config.optimization,
        ever_config.model,
        ever_config.environment.random_seed,
        ever_config.optimization_output_dir,
    )
    assert len(ropt_config["variables"]["lower_bounds"]) == 6
    assert len(ropt_config["variables"]["upper_bounds"]) == 6


def test_everest2ropt_controls_input_constraint(ever_config):
    input_constraints_ever_config = ever_config.input_constraints
    assert len(input_constraints_ever_config) == 2
    ropt_config, _ = everest2ropt(
        ever_config.controls,
        ever_config.objective_functions,
        ever_config.input_constraints,
        ever_config.output_constraints,
        ever_config.optimization,
        ever_config.model,
        ever_config.environment.random_seed,
        ever_config.optimization_output_dir,
    )
    assert len(ropt_config["linear_constraints"]["coefficients"]) == 2
    exp_lower_bounds = [0.0, 1.0]
    exp_upper_bounds = [1.0, 1.0]
    assert exp_lower_bounds == ropt_config["linear_constraints"]["lower_bounds"]
    assert exp_upper_bounds == ropt_config["linear_constraints"]["upper_bounds"]


def test_everest2ropt_controls_optimizer_setting(ever_config):
    ropt_config, _ = everest2ropt(
        ever_config.controls,
        ever_config.objective_functions,
        ever_config.input_constraints,
        ever_config.output_constraints,
        ever_config.optimization,
        ever_config.model,
        ever_config.environment.random_seed,
        ever_config.optimization_output_dir,
    )
    assert len(ropt_config["realizations"]["weights"]) == 2
    assert ropt_config["optimizer"]["method"] == "optpp_q_newton"
    assert ropt_config["gradient"]["number_of_perturbations"] == 2


def test_everest2ropt_constraints(ever_config):
    ropt_config, _ = everest2ropt(
        ever_config.controls,
        ever_config.objective_functions,
        ever_config.input_constraints,
        ever_config.output_constraints,
        ever_config.optimization,
        ever_config.model,
        ever_config.environment.random_seed,
        ever_config.optimization_output_dir,
    )
    assert len(ropt_config["nonlinear_constraints"]["lower_bounds"]) == 2


def test_everest2ropt_backend_options(ever_config):
    ever_config.optimization.options = ["test = 1"]
    ropt_config, _ = everest2ropt(
        ever_config.controls,
        ever_config.objective_functions,
        ever_config.input_constraints,
        ever_config.output_constraints,
        ever_config.optimization,
        ever_config.model,
        ever_config.environment.random_seed,
        ever_config.optimization_output_dir,
    )
    assert ropt_config["optimizer"]["options"] == ["test = 1"]

    ever_config.optimization.backend_options = {"test": "5"}  # should be disregarded
    ropt_config, _ = everest2ropt(
        ever_config.controls,
        ever_config.objective_functions,
        ever_config.input_constraints,
        ever_config.output_constraints,
        ever_config.optimization,
        ever_config.model,
        ever_config.environment.random_seed,
        ever_config.optimization_output_dir,
    )
    assert ropt_config["optimizer"]["options"] == ["test = 1"]

    ever_config.optimization.options = None
    ropt_config, _ = everest2ropt(
        ever_config.controls,
        ever_config.objective_functions,
        ever_config.input_constraints,
        ever_config.output_constraints,
        ever_config.optimization,
        ever_config.model,
        ever_config.environment.random_seed,
        ever_config.optimization_output_dir,
    )
    assert ropt_config["optimizer"]["options"] == {"test": "5"}

    ever_config.optimization.options = ["hey", "a=b", "c 100"]
    ropt_config, _ = everest2ropt(
        ever_config.controls,
        ever_config.objective_functions,
        ever_config.input_constraints,
        ever_config.output_constraints,
        ever_config.optimization,
        ever_config.model,
        ever_config.environment.random_seed,
        ever_config.optimization_output_dir,
    )
    assert ropt_config["optimizer"]["options"] == ["hey", "a=b", "c 100"]


def test_everest2ropt_samplers(ever_config):
    ropt_config, _ = everest2ropt(
        ever_config.controls,
        ever_config.objective_functions,
        ever_config.input_constraints,
        ever_config.output_constraints,
        ever_config.optimization,
        ever_config.model,
        ever_config.environment.random_seed,
        ever_config.optimization_output_dir,
    )

    assert len(ropt_config["samplers"]) == 3
    assert ropt_config["variables"]["samplers"] == [0, 1, 2, 0, 0, 0]
    assert ropt_config["samplers"][0]["method"] == "norm"
    assert not ropt_config["samplers"][0]["shared"]
    assert ropt_config["samplers"][1]["method"] == "uniform"
    assert not ropt_config["samplers"][1]["shared"]
    assert ropt_config["samplers"][2]["method"] == "norm"
    assert ropt_config["samplers"][2]["shared"]


def test_everest2ropt_cvar(ever_config):
    config_dict = ever_config.to_dict()

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

    config = EverestConfig.model_validate(config_dict)
    ropt_config, _ = everest2ropt(
        config.controls,
        config.objective_functions,
        config.input_constraints,
        config.output_constraints,
        config.optimization,
        config.model,
        config.environment.random_seed,
        config.optimization_output_dir,
    )

    assert ropt_config["objectives"]["realization_filters"] == [0]
    assert len(ropt_config["realization_filters"]) == 1
    assert ropt_config["realization_filters"][0]["method"] == "sort-objective"
    assert ropt_config["realization_filters"][0]["options"]["sort"] == [0]
    assert ropt_config["realization_filters"][0]["options"]["first"] == 0
    assert ropt_config["realization_filters"][0]["options"]["last"] == 0

    config_dict["optimization"]["cvar"] = {
        "percentile": 0.3,
    }

    config = EverestConfig.model_validate(config_dict)
    ropt_config, _ = everest2ropt(
        config.controls,
        config.objective_functions,
        config.input_constraints,
        config.output_constraints,
        config.optimization,
        config.model,
        config.environment.random_seed,
        config.optimization_output_dir,
    )
    assert ropt_config["objectives"]["realization_filters"] == [0]
    assert len(ropt_config["realization_filters"]) == 1
    assert ropt_config["realization_filters"][0]["method"] == "cvar-objective"
    assert ropt_config["realization_filters"][0]["options"]["sort"] == [0]
    assert ropt_config["realization_filters"][0]["options"]["percentile"] == 0.3


def test_everest2ropt_arbitrary_backend_options(ever_config):
    ever_config.optimization.backend_options = {"a": [1]}
    ropt_config, _ = everest2ropt(
        ever_config.controls,
        ever_config.objective_functions,
        ever_config.input_constraints,
        ever_config.output_constraints,
        ever_config.optimization,
        ever_config.model,
        ever_config.environment.random_seed,
        ever_config.optimization_output_dir,
    )
    assert "a" in ropt_config["optimizer"]["options"]
    assert ropt_config["optimizer"]["options"]["a"] == [1]


def test_everest2ropt_default_algorithm_name(min_config):
    config = EverestConfig(**min_config)
    assert not min_config.get("optimization")
    ropt_config, _ = everest2ropt(
        config.controls,
        config.objective_functions,
        config.input_constraints,
        config.output_constraints,
        config.optimization,
        config.model,
        config.environment.random_seed,
        config.optimization_output_dir,
    )
    assert ropt_config["optimizer"]["method"] == "optpp_q_newton"


@pytest.mark.parametrize(
    "case", ["config_advanced.yml", "config_multiobj.yml", "config_minimal.yml"]
)
def test_everest2ropt_snapshot(case, snapshot):
    config = EverestConfig.load_file(
        relpath(f"../../test-data/everest/math_func/{case}")
    )
    ropt_config_dict, _ = everest2ropt(
        config.controls,
        config.objective_functions,
        config.input_constraints,
        config.output_constraints,
        config.optimization,
        config.model,
        config.environment.random_seed,
        config.optimization_output_dir,
    )
    ropt_config_dict["optimizer"]["output_dir"] = "not_relevant"

    ropt_config_str = (
        orjson.dumps(
            ropt_config_dict,
            option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS,
        )
        .decode("utf-8")
        .strip()
        + "\n"
    )
    snapshot.assert_match(ropt_config_str, "ropt_config.json")


def test_everest2ropt_validation_error(
    change_to_tmpdir, ever_config, monkeypatch
) -> None:
    def _patched_everest2ropt(*args, **kwargs):
        ropt_dict, initial_value = everest2ropt(*args, **kwargs)
        ropt_dict["foo"] = "bar"
        return ropt_dict, initial_value

    runtime_plugins = get_site_plugins()
    run_model = EverestRunModel.create(ever_config, runtime_plugins=runtime_plugins)

    monkeypatch.setattr(everest_run_model, "everest2ropt", _patched_everest2ropt)
    evaluator_server_config = EvaluatorServerConfig()
    with pytest.raises(ValueError, match=r"Validation error\(s\) in ropt"):
        run_model.run_experiment(evaluator_server_config)
