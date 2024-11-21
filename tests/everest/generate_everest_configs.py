import os
from typing import Any, Dict, List, Tuple

from everest.config import (
    ControlConfig,
    EverestConfig,
)


# @pytest.fixture
def generate_controls_config(
    name: str = "point",
    type: str = "generic_control",
    variables: Tuple[Dict[str, str | int | float]] = (
        {"name": "x"},
        {"name": "y"},
        {"name": "z"},
    ),
    min: float | None = -1.0,
    max: float | None = 1.0,
    initial_guess: float | None = 0.1,
    pertubation_magnitude: float | None = 0.001,
    control_type: str | None = None,
    # ) -> ControlConfig:
) -> Dict[str, str | int | List[Any] | float]:
    config = {
        "name": name,
        "type": type,
        "min": min,
        "max": max,
        "initial_guess": initial_guess,
        "perturbation_magnitude": pertubation_magnitude,
        "variables": variables,
        "control_type": control_type,
    }
    # return ControlConfig.model_validate(_extract_non_none_from_dict(config))
    return _extract_non_none_from_dict(config)


# @pytest.fixture
def generate_objective_function_config(
    name: str = "distance",
    # ) -> ObjectiveFunctionConfig:
) -> Dict[str, str | int | List[Any] | float]:
    config = {"name": name}
    # return ObjectiveFunctionConfig.model_validate(_extract_non_none_from_dict(config))
    return _extract_non_none_from_dict(config)


# @pytest.fixture
def generate_optimization_config(
    algorithm: str | None = "optpp_q_newton",
    convergence_tolerance: float | None = 0.001,
    max_batch_num: int | None = 4,
    constraint_tolerance: float | None = 0.1,
    perturbation_num: int | None = None,
    speculative: bool | None = None,
    # ) -> OptimizationConfig:
) -> Dict[str, str | int | List[Any] | float]:
    config = {
        "algorithm": algorithm,
        "convergence_tolerance": convergence_tolerance,
        "max_batch_num": max_batch_num,
        "constraint_tolerance": constraint_tolerance,
        "perturbation_num": perturbation_num,
        "speculative": speculative,
    }
    # return OptimizationConfig.model_validate(_extract_non_none_from_dict(config))
    return _extract_non_none_from_dict(config)


# @pytest.fixture
def generate_install_jobs_config(
    name: str = "distance3",
    source: str = "jobs/DISTANCE3",
    # ) -> InstallJobConfig:
) -> Dict[str, str | int | List[Any] | float]:
    config = {
        "name": name,
        "source": source,
    }
    # return InstallJobConfig.model_validate(_extract_non_none_from_dict(config))
    return _extract_non_none_from_dict(config)


# @pytest.fixture
def generate_model_config(
    realizations: int = 0,
    realizations_weights: List[float] | None = None,
    # ) -> ModelConfig:
) -> Dict[str, str | int | List[Any] | float]:
    config = {
        "realizations": [realizations],
        "realizations_weights": realizations_weights,
    }
    # return ModelConfig.model_validate(_extract_non_none_from_dict(config))
    return _extract_non_none_from_dict(config)


# @pytest.fixture
def generate_forward_model_config(
    forward_models: str = "distance3 --point-file point.json --target 0.5 0.5 0.5 --out distance",
) -> List[str]:
    return [forward_models]


# @pytest.fixture
def generate_environment_config(
    simulation_folder: str | None = "sim_output",
    log_level: str | None = "debug",
    random_seed: int | None = 123,
    output_folder: str | None = None,
    # ) -> EnvironmentConfig:
) -> Dict[str, str | int | List[Any] | float]:
    config = {
        "simulation_folder": simulation_folder,
        "log_level": log_level,
        "random_seed": random_seed,
        "output_folder": output_folder,
    }
    # return EnvironmentConfig.model_validate(_extract_non_none_from_dict(config))
    return _extract_non_none_from_dict(config)


# @pytest.fixture
def generate_minimal_everest_config() -> EverestConfig:
    everest_config = {}
    everest_config["controls"] = [generate_controls_config()]
    everest_config["objective_functions"] = [generate_objective_function_config()]
    everest_config["optimization"] = generate_optimization_config()
    everest_config["model"] = generate_model_config()
    everest_config["environment"] = generate_environment_config()
    everest_config["install_jobs"] = [generate_install_jobs_config()]
    everest_config["forward_model"] = generate_forward_model_config()
    everest_config["config_path"] = os.path.abspath(
        os.path.join("./test-data", "everest", "math_func", "config_minimal.yml")
    )
    return EverestConfig.model_validate(everest_config)


# @pytest.fixture
def generate_controls_advanced() -> ControlConfig:
    config = generate_controls_config(
        initial_guess=0.25,
        pertubation_magnitude=0.005,
        variables=[
            {"name": "x", "index": 0},
            {"name": "x", "index": 1},
            {"name": "x", "index": 2},
        ],
    )
    return config


# @pytest.fixture
def generate_controls_auto_scaled_controls() -> ControlConfig:
    config = generate_controls_config(initial_guess=0.2)
    config.auto_scale = True
    config.scaled_range = (0.3, 0.7)
    return config


# @pytest.fixture
def generate_controls_cvar() -> ControlConfig:
    config = generate_controls_config(
        initial_guess=None,
        pertubation_magnitude=0.01,
        variables=[
            {"name": "x", "initial_guess": 0.0},
            {"name": "y", "initial_guess": 0.0},
            {"name": "z", "initial_guess": 0.0},
        ],
        min=-2.0,
        max=2.0,
    )
    return config


# @pytest.fixture
def generate_controls_fm_failure() -> ControlConfig:
    return generate_controls_config()


# @pytest.fixture
def generate_controls_minimal_slow() -> ControlConfig:
    return generate_controls_config()


def _extract_non_none_from_dict(config: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in config.items() if value is not None}
