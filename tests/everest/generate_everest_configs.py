import os
from typing import Any, Dict, List, Optional


def generate_default_minimal_config_dict() -> Dict[str, Any]:
    everest_config = {}
    everest_config["controls"] = [generate_controls_config()]
    everest_config["objective_functions"] = [generate_objective_function_config()]
    everest_config["optimization"] = generate_optimization_config(
        constraint_tolerance=0.1
    )
    everest_config["model"] = generate_model_config(realizations=[0])
    everest_config["environment"] = generate_environment_config()
    everest_config["install_jobs"] = [generate_install_jobs_config()]
    everest_config["forward_model"] = [
        "distance3 --point-file point.json --target 0.5 0.5 0.5 --out distance"
    ]
    everest_config["config_path"] = os.path.abspath(
        os.path.join("./test-data", "everest", "math_func", "config_minimal.yml")
    )
    return everest_config


def generate_default_advanced_config_dict() -> Dict[str, Any]:
    everest_config = generate_default_minimal_config_dict()
    variables = [
        {"name": "x", "index": 0},
        {"name": "x", "index": 1},
        {"name": "x", "index": 2},
    ]
    everest_config["controls"] = [
        generate_controls_config(
            variables=variables, initial_guess=0.25, perturbation_magnitude=0.005
        )
    ]

    everest_config["optimization"] = generate_optimization_config(
        convergence_tolerance=0.005,
        perturbation_num=7,
        speculative=True,
        max_batch_num=None,
    )

    everest_config["model"] = generate_model_config(
        realizations=[0, 2], realizations_weights=[0.25, 0.75]
    )

    everest_config["environment"] = generate_environment_config(
        simulation_folder="scratch/advanced/",
        output_folder="everest_output/",
        random_seed=999,
    )

    abs_config_path = os.path.abspath(
        os.path.join("./test-data", "everest", "math_func")
    )
    install_data = {
        "link": False,
        "source": f"{abs_config_path}/adv_target_<GEO_ID>.json",
        "target": "data/<GEO_ID>/target.json",
    }
    everest_config["install_data"] = [install_data]
    everest_config["install_templates"] = []
    everest_config["wells"] = []

    everest_config["input_constraints"] = [
        generate_input_constraints_config(
            weights={"point.x-0": 0, "point.x-1": 0, "point.x-2": 1}
        )
    ]
    everest_config["output_constraints"] = [
        {"name": "x-0_coord", "lower_bound": 0.1, "scale": 0.1}
    ]

    everest_config["install_jobs"] = [
        generate_install_jobs_config(name="adv_distance3", source="jobs/ADV_DISTANCE3"),
        generate_install_jobs_config(
            name="adv_dump_controls", source="jobs/ADV_DUMP_CONTROLS"
        ),
    ]

    everest_config["forward_model"] = [
        "adv_distance3     --point-file point.json --target-file data/<GEO_ID>/target.json --out distance",
        "adv_dump_controls --controls-file point.json --out-suffix _coord",
    ]

    everest_config["config_path"] = os.path.join(abs_config_path, "config_advanced.yml")
    return everest_config


def generate_controls_config(
    name: str = "point",
    type: str = "generic_control",
    variables: Optional[List[Dict[str, str | int | float]]] = None,
    min: Optional[float] = -1.0,
    max: Optional[float] = 1.0,
    initial_guess: Optional[float] = 0.1,
    perturbation_magnitude: Optional[float] = 0.001,
    control_type: Optional[str] = None,
    auto_scale: Optional[bool] = None,
    scaled_range: Optional[List[float | int]] = None,
) -> Dict[str, str | int | List[Any] | float]:
    if variables is None:
        variables = [
            {"name": "x"},
            {"name": "y"},
            {"name": "z"},
        ]

    config = {
        "name": name,
        "type": type,
        "min": min,
        "max": max,
        "initial_guess": initial_guess,
        "perturbation_magnitude": perturbation_magnitude,
        "variables": variables,
        "control_type": control_type,
        "auto_scale": auto_scale,
        "scaled_range": scaled_range,
    }
    return _extract_non_none_from_dict(config)


def generate_objective_function_config(
    name: str = "distance",
    weight: Optional[float] = None,
    normalization: Optional[float] = None,
    type: Optional[str] = None,
    alias: Optional[str] = None,
) -> Dict[str, str | int | List[Any] | float]:
    config = {
        "name": name,
        "weight": weight,
        "normalization": normalization,
        "type": type,
        "alias": alias,
    }
    return _extract_non_none_from_dict(config)


def generate_optimization_config(
    algorithm: Optional[str] = "optpp_q_newton",
    convergence_tolerance: Optional[float] = 0.001,
    max_batch_num: Optional[int] = 4,
    constraint_tolerance: Optional[float] = None,
    perturbation_num: Optional[int] = None,
    speculative: Optional[bool] = None,
    backend: Optional[str] = None,
    cvar: Optional[Dict[str, int | float | str]] = None,
    min_realizations_success: Optional[int] = None,
    min_pert_success: Optional[int] = None,
    max_iterations: Optional[int] = None,
    backend_options: Optional[Dict[str, int]] = None,
) -> Dict[str, str | int | List[Any] | float]:
    config = {
        "algorithm": algorithm,
        "convergence_tolerance": convergence_tolerance,
        "max_batch_num": max_batch_num,
        "constraint_tolerance": constraint_tolerance,
        "perturbation_num": perturbation_num,
        "speculative": speculative,
        "backend": backend,
        "cvar": cvar,
        "min_realizations_success": min_realizations_success,
        "min_pert_success": min_pert_success,
        "max_iterations": max_iterations,
        "backend_options": backend_options,
    }
    return _extract_non_none_from_dict(config)


def generate_install_jobs_config(
    name: str = "distance3",
    source: str = "jobs/DISTANCE3",
) -> Dict[str, str | int | List[Any] | float]:
    config = {
        "name": name,
        "source": source,
    }
    return _extract_non_none_from_dict(config)


def generate_model_config(
    realizations: Optional[List[int]] = None,
    realizations_weights: Optional[List[float]] = None,
) -> Dict[str, str | int | List[Any] | float]:
    if realizations is None:
        realizations = [0]

    config = {
        "realizations": realizations,
        "realizations_weights": realizations_weights,
    }
    return _extract_non_none_from_dict(config)


def generate_input_constraints_config(
    weights: Dict[str, int | float],
    upper_bound: float = 0.4,
) -> Dict[str, Dict[str, int | float] | int | float]:
    config = {
        "weights": weights,
        "upper_bound": upper_bound,
    }
    return config


def generate_environment_config(
    simulation_folder: Optional[str] = "sim_output",
    log_level: Optional[str] = "debug",
    random_seed: Optional[int] = 123,
    output_folder: Optional[str] = None,
) -> Dict[str, str | int | List[Any] | float]:
    config = {
        "simulation_folder": simulation_folder,
        "log_level": log_level,
        "random_seed": random_seed,
        "output_folder": output_folder,
    }
    return _extract_non_none_from_dict(config)


def _extract_non_none_from_dict(config: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in config.items() if value is not None}
