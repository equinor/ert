import os
from typing import Any, Dict, List, Optional

from everest.config import EverestConfig


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


def generate_minimal_everest_config_file() -> EverestConfig:
    return EverestConfig.model_validate(generate_default_minimal_config_dict())


def generate_auto_scaled_controls_everest_config_file() -> EverestConfig:
    everest_config = generate_default_minimal_config_dict()
    everest_config["controls"] = [
        generate_controls_config(
            initial_guess=0.2, auto_scale=True, scaled_range=[0.3, 0.7]
        )
    ]
    everest_config["optimization"] = generate_optimization_config(max_batch_num=10)
    everest_config["environment"] = generate_environment_config(random_seed=999)
    everest_config["forward_model"] = [
        "distance3 --point-file point.json --target 0.5 0.5 0.5 --out distance --scaling -1 1 0.3 0.7"
    ]
    everest_config["input_constraints"] = [
        generate_input_constraints_config(
            weights={"point.x": 1.0, "point.y": 1.0}, upper_bound=0.5
        )
    ]
    everest_config["config_path"] = os.path.abspath(
        os.path.join(
            "./test-data", "everest", "math_func", "config_auto_scaled_controls.yml"
        )
    )
    return EverestConfig.model_validate(everest_config)


def generate_cvar_everest_config_file() -> EverestConfig:
    everest_config = generate_default_minimal_config_dict()
    variables = [
        {"name": "x", "initial_guess": 0.0},
        {"name": "y", "initial_guess": 0.0},
        {"name": "z", "initial_guess": 0.0},
    ]
    everest_config["controls"] = [
        generate_controls_config(
            perturbation_magnitude=0.01,
            min=-2.0,
            max=2.0,
            variables=variables,
            initial_guess=None,
        )
    ]
    everest_config["optimization"] = generate_optimization_config(
        max_batch_num=5,
        backend="scipy",
        algorithm="slsqp",
        cvar={"percentile": 0.5},
        convergence_tolerance=None,
    )
    everest_config["model"] = generate_model_config(realizations=[0, 1])
    everest_config["environment"] = generate_environment_config(
        random_seed=999, output_folder="distance_output", log_level="info"
    )
    everest_config["forward_model"] = [
        "distance3 --point-file point.json --realization <GEO_ID> --target 0.5 0.5 0.5 --out distance"
    ]
    everest_config["config_path"] = os.path.abspath(
        os.path.join("./test-data", "everest", "math_func", "config_cvar.yml")
    )
    return EverestConfig.model_validate(everest_config)


def generate_advanced_everest_config_file() -> EverestConfig:
    return EverestConfig.model_validate(generate_default_advanced_config_dict())


def generate_advanced_scipy_everest_config_file() -> EverestConfig:
    everest_config = generate_default_advanced_config_dict()
    everest_config["optimization"] = generate_optimization_config(
        convergence_tolerance=0.001,
        constraint_tolerance=0.001,
        backend="scipy",
        algorithm="SLSQP",
        speculative=True,
        max_batch_num=4,
        perturbation_num=7,
        backend_options={"maxiter": 100},
    )
    everest_config["environment"] = generate_environment_config(
        simulation_folder="scratch/advanced/", output_folder="everest_output/"
    )

    everest_config["config_path"] = os.path.abspath(
        os.path.join("./test-data", "everest", "math_func", "config_advanced_scipy.yml")
    )
    return EverestConfig.model_validate(everest_config)


def generate_minimal_slow_everest_config_file() -> EverestConfig:
    everest_config = generate_default_minimal_config_dict()
    everest_config["optimization"] = generate_optimization_config()
    everest_config["install_jobs"] = [
        generate_install_jobs_config(name="distance3", source="jobs/DISTANCE3"),
        generate_install_jobs_config(name="sleep", source="jobs/SLEEP"),
    ]
    everest_config["forward_model"] = [
        "distance3 --point-file point.json --target 0.5 0.5 0.5 --out distance",
        "sleep --sleep 10",
    ]
    everest_config["config_path"] = os.path.abspath(
        os.path.join("./test-data", "everest", "math_func", "config_minimal_slow.yml")
    )
    return EverestConfig.model_validate(everest_config)


def generate_multiobj_everest_config_file() -> EverestConfig:
    everest_config = generate_default_minimal_config_dict()
    everest_config["controls"] = [
        generate_controls_config(initial_guess=0, perturbation_magnitude=0.01)
    ]
    everest_config["objective_functions"] = [
        generate_objective_function_config(
            name="distance_p", weight=0.5, normalization=1.5
        ),
        generate_objective_function_config(
            name="distance_q", weight=0.25, normalization=1.0
        ),
    ]
    everest_config["optimization"] = generate_optimization_config(
        convergence_tolerance=0.005, perturbation_num=5, max_batch_num=3
    )
    everest_config["environment"] = generate_environment_config(
        output_folder="everest_output_multiobj", random_seed=999
    )
    everest_config["forward_model"] = [
        "distance3 --point-file point.json --target 0.5 0.5 0.5 --out distance_p",
        "distance3 --point-file point.json --target -1.5 -1.5 0.5 --out distance_q",
    ]
    everest_config["config_path"] = os.path.abspath(
        os.path.join("./test-data", "everest", "math_func", "config_multiobj.yml")
    )
    return EverestConfig.model_validate(everest_config)


def generate_one_batch_everest_config_file() -> EverestConfig:
    everest_config = generate_default_minimal_config_dict()
    everest_config["optimization"] = generate_optimization_config(
        convergence_tolerance=None,
        max_batch_num=1,
    )
    everest_config["environment"] = generate_environment_config(random_seed=999)

    everest_config["config_path"] = os.path.abspath(
        os.path.join("./test-data", "everest", "math_func", "config_one_batch.yml")
    )
    return EverestConfig.model_validate(everest_config)


def generate_remove_run_path_everest_config_file() -> EverestConfig:
    everest_config = generate_default_minimal_config_dict()
    everest_config["controls"] = [generate_controls_config(initial_guess=0)]
    everest_config["optimization"] = generate_optimization_config(
        convergence_tolerance=0.005,
        max_batch_num=None,
        min_realizations_success=1,
        min_pert_success=1,
        max_iterations=1,
        perturbation_num=2,
    )
    everest_config["environment"] = generate_environment_config(
        output_folder="everest_output/", simulation_folder="scratch/advanced/"
    )
    everest_config["install_jobs"] = [
        generate_install_jobs_config(name="distance3", source="jobs/DISTANCE3"),
        generate_install_jobs_config(
            name="toggle_failure", source="jobs/FAIL_SIMULATION"
        ),
    ]
    everest_config["forward_model"] = [
        "distance3 --point-file point.json --target 0.5 0.5 0.5 --out distance",
        "toggle_failure",
    ]
    everest_config["config_path"] = os.path.abspath(
        os.path.join(
            "./test-data", "everest", "math_func", "config_remove_run_path.yml"
        )
    )
    everest_config["simulator"] = {"delete_run_path": True}
    return EverestConfig.model_validate(everest_config)


def generate_stddev_everest_config_file() -> EverestConfig:
    everest_config = generate_default_minimal_config_dict()
    variables = [
        {"name": "x", "initial_guess": 0.0},
        {"name": "y", "initial_guess": 0.0},
        {"name": "z", "initial_guess": 0.0},
    ]
    everest_config["controls"] = [
        generate_controls_config(
            perturbation_magnitude=0.01, variables=variables, initial_guess=None
        )
    ]
    everest_config["objective_functions"] = [
        generate_objective_function_config(name="distance", weight=1.0),
        generate_objective_function_config(
            name="stddev", weight=1.0, type="stddev", alias="distance"
        ),
    ]
    everest_config["optimization"] = generate_optimization_config(
        max_batch_num=5,
        backend="scipy",
        algorithm="slsqp",
        convergence_tolerance=0.0001,
        perturbation_num=3,
    )
    everest_config["model"] = generate_model_config(realizations=[0, 1])
    everest_config["environment"] = generate_environment_config(
        random_seed=999, output_folder="distance_output", log_level=None
    )
    everest_config["forward_model"] = [
        "distance3 --point-file point.json --realization <GEO_ID> --target 0.5 0.5 0.5 --out distance"
    ]
    everest_config["config_path"] = os.path.abspath(
        os.path.join("./test-data", "everest", "math_func", "config_stddev.yml")
    )
    return EverestConfig.model_validate(everest_config)


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
