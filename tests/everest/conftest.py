import os
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Callable, Dict, Iterator, Optional, Union

import pytest

from ert.config import QueueSystem
from ert.ensemble_evaluator import EvaluatorServerConfig
from everest.config import EverestConfig
from everest.config.control_config import ControlConfig
from tests.everest.generate_everest_configs import (
    generate_controls_config,
    generate_default_advanced_config_dict,
    generate_default_minimal_config_dict,
    generate_environment_config,
    generate_input_constraints_config,
    generate_install_jobs_config,
    generate_model_config,
    generate_objective_function_config,
    generate_optimization_config,
)
from tests.everest.utils import relpath


@pytest.fixture(scope="session")
def testdata() -> Path:
    return Path(__file__).parent / "test_data"


@pytest.fixture
def copy_testdata_tmpdir(
    testdata: Path, tmp_path: Path
) -> Iterator[Callable[[Optional[str]], Path]]:
    def _copy_tree(path: Optional[str] = None):
        path_ = testdata if path is None else testdata / path
        shutil.copytree(path_, tmp_path, dirs_exist_ok=True)
        return path_

    cwd = Path.cwd()
    os.chdir(tmp_path)
    yield _copy_tree
    os.chdir(cwd)


@pytest.fixture(scope="module")
def control_data_no_variables() -> Dict[str, Union[str, float]]:
    return {
        "name": "group_0",
        "type": "well_control",
        "min": 0.0,
        "max": 0.1,
        "perturbation_magnitude": 0.005,
    }


@pytest.fixture(
    scope="module",
    params=(
        pytest.param(
            [
                {"name": "w00", "initial_guess": 0.0626, "index": 0},
                {"name": "w00", "initial_guess": 0.063, "index": 1},
                {"name": "w00", "initial_guess": 0.0617, "index": 2},
                {"name": "w00", "initial_guess": 0.0621, "index": 3},
                {"name": "w01", "initial_guess": 0.0627, "index": 0},
                {"name": "w01", "initial_guess": 0.0631, "index": 1},
                {"name": "w01", "initial_guess": 0.0618, "index": 2},
                {"name": "w01", "initial_guess": 0.0622, "index": 3},
                {"name": "w02", "initial_guess": 0.0628, "index": 0},
                {"name": "w02", "initial_guess": 0.0632, "index": 1},
                {"name": "w02", "initial_guess": 0.0619, "index": 2},
                {"name": "w02", "initial_guess": 0.0623, "index": 3},
                {"name": "w03", "initial_guess": 0.0629, "index": 0},
                {"name": "w03", "initial_guess": 0.0633, "index": 1},
                {"name": "w03", "initial_guess": 0.062, "index": 2},
                {"name": "w03", "initial_guess": 0.0624, "index": 3},
            ],
            id="indexed variables",
        ),
        pytest.param(
            [
                {"name": "w00", "initial_guess": [0.0626, 0.063, 0.0617, 0.0621]},
                {"name": "w01", "initial_guess": [0.0627, 0.0631, 0.0618, 0.0622]},
                {"name": "w02", "initial_guess": [0.0628, 0.0632, 0.0619, 0.0623]},
                {"name": "w03", "initial_guess": [0.0629, 0.0633, 0.062, 0.0624]},
            ],
            id="vectored variables",
        ),
    ),
)
def control_config(
    request,
    control_data_no_variables: Dict[str, Union[str, float]],
) -> ControlConfig:
    config = deepcopy(control_data_no_variables)
    config["variables"] = request.param
    return ControlConfig.model_validate(config)


@pytest.fixture
def copy_math_func_test_data_to_tmp(tmp_path, monkeypatch):
    path = relpath("..", "..", "test-data", "everest", "math_func")
    shutil.copytree(path, tmp_path, dirs_exist_ok=True)
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def copy_mocked_test_data_to_tmp(tmp_path, monkeypatch):
    path = relpath("test_data", "mocked_test_case")
    shutil.copytree(path, tmp_path, dirs_exist_ok=True)
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def copy_test_data_to_tmp(tmp_path, monkeypatch):
    path = relpath("test_data")
    shutil.copytree(path, tmp_path, dirs_exist_ok=True)
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def copy_template_test_data_to_tmp(tmp_path, monkeypatch):
    path = relpath("test_data", "templating")
    shutil.copytree(path, tmp_path, dirs_exist_ok=True)
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def copy_egg_test_data_to_tmp(tmp_path, monkeypatch):
    path = relpath("..", "..", "test-data", "everest", "egg")
    shutil.copytree(path, tmp_path, dirs_exist_ok=True)
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def change_to_tmpdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def evaluator_server_config_generator():
    def create_evaluator_server_config(run_model):
        return EvaluatorServerConfig(
            custom_port_range=range(49152, 51819)
            if run_model.ert_config.queue_config.queue_system == QueueSystem.LOCAL
            else None
        )

    return create_evaluator_server_config


@pytest.fixture
def minimal_everest_config() -> EverestConfig:
    return EverestConfig.model_validate(generate_default_minimal_config_dict())


@pytest.fixture
def auto_scaled_controls_everest_config() -> EverestConfig:
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


@pytest.fixture
def cvar_everest_config() -> EverestConfig:
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


@pytest.fixture
def advanced_everest_config() -> EverestConfig:
    return EverestConfig.model_validate(generate_default_advanced_config_dict())


@pytest.fixture
def advanced_scipy_everest_config() -> EverestConfig:
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


@pytest.fixture
def minimal_slow_everest_config() -> EverestConfig:
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


@pytest.fixture
def multiobj_everest_config() -> EverestConfig:
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


@pytest.fixture
def one_batch_everest_config() -> EverestConfig:
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


@pytest.fixture
def remove_run_path_everest_config() -> EverestConfig:
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


@pytest.fixture
def stddev_everest_config() -> EverestConfig:
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
