import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from ert.ensemble_evaluator.config import EvaluatorServerConfig
from ert.run_models.everest_run_model import EverestRunModel
from everest.config import EverestConfig
from everest.everest_storage import EverestStorage
from everest.util import makedirs_if_needed

CONFIG_FILE_MULTIOBJ = "config_multiobj.yml"
CONFIG_FILE_ADVANCED = "config_advanced.yml"


@pytest.mark.integration_test
def test_math_func_multiobj(cached_example):
    config_path, config_file, _ = cached_example("math_func/config_multiobj.yml")

    config = EverestConfig.load_file(Path(config_path) / config_file)

    storage = EverestStorage(Path(config.optimization_output_dir))
    storage.read_from_output_dir()
    result = storage.get_optimal_result()
    # Check resulting points
    x, y, z = (result.controls["point." + p] for p in ("x", "y", "z"))
    assert x == pytest.approx(0.0, abs=0.05)
    assert y == pytest.approx(0.0, abs=0.05)
    assert z == pytest.approx(0.5, abs=0.05)

    # The overall optimum is a weighted average of the objectives
    assert result.total_objective == pytest.approx(
        (-0.5 * (2.0 / 3.0) * 1.5) + (-4.5 * (1.0 / 3.0) * 1.0), abs=0.01
    )


@pytest.mark.integration_test
def test_math_func_advanced(cached_example):
    config_path, config_file, _ = cached_example("math_func/config_advanced.yml")

    config = EverestConfig.load_file(Path(config_path) / config_file)
    storage = EverestStorage(Path(config.optimization_output_dir))
    storage.read_from_output_dir()
    result = storage.get_optimal_result()

    point_names = ["x.0", "x.1", "x.2"]
    # Check resulting points
    x0, x1, x2 = (result.controls["point." + p] for p in point_names)
    assert x0 == pytest.approx(0.1, abs=0.05)
    assert x1 == pytest.approx(0.0, abs=0.05)
    assert x2 == pytest.approx(0.4, abs=0.05)

    # Check optimum value
    assert pytest.approx(result.total_objective, abs=0.1) == -(
        0.25 * (1.6**2 + 1.5**2 + 0.1**2) + 0.75 * (0.4**2 + 0.5**2 + 0.1**2)
    )
    # Expected distance is the weighted average of the (squared) distances
    #  from (x, y, z) to (-1.5, -1.5, 0.5) and (0.5, 0.5, 0.5)
    w = config.model.realizations_weights
    assert w == [0.25, 0.75]
    dist_0 = (x0 + 1.5) ** 2 + (x1 + 1.5) ** 2 + (x2 - 0.5) ** 2
    dist_1 = (x0 - 0.5) ** 2 + (x1 - 0.5) ** 2 + (x2 - 0.5) ** 2
    expected_opt = -(w[0] * (dist_0) + w[1] * (dist_1))
    assert expected_opt == pytest.approx(result.total_objective, abs=0.001)


@pytest.mark.integration_test
def test_remove_run_path(copy_math_func_test_data_to_tmp):
    with open("config_minimal.yml", encoding="utf-8") as file:
        config_yaml = yaml.safe_load(file)
        config_yaml["simulator"] = {"delete_run_path": True}
        config_yaml["install_jobs"].append(
            {"name": "toggle_failure", "source": "jobs/FAIL_SIMULATION"}
        )
        config_yaml["forward_model"].append("toggle_failure --fail simulation_2")
    with open("config.yml", "w", encoding="utf-8") as fout:
        yaml.dump(config_yaml, fout)
    config = EverestConfig.load_file("config.yml")

    simulation_dir = config.simulation_dir

    run_model = EverestRunModel.create(config)
    evaluator_server_config = EvaluatorServerConfig()
    run_model.run_experiment(evaluator_server_config)

    # Check the failed simulation folder still exists
    assert os.path.exists(
        os.path.join(simulation_dir, "batch_0/geo_realization_0/simulation_2")
    ), "Simulation folder should be there, something went wrong and was removed!"

    # Check the successful simulation folders do not exist
    assert not os.path.exists(
        os.path.join(simulation_dir, "batch_0/geo_realization_0/simulation_0")
    ), "Simulation folder should not be there, something went wrong!"

    assert not os.path.exists(
        os.path.join(simulation_dir, "batch_0/geo_realization_0/simulation_1")
    ), "Simulation folder should not be there, something went wrong!"

    # Manually rolling the output folder between two runs
    makedirs_if_needed(config.output_dir, roll_if_exists=True)

    config.simulator.delete_run_path = False
    run_model = EverestRunModel.create(config)
    evaluator_server_config = EvaluatorServerConfig()
    run_model.run_experiment(evaluator_server_config)

    # Check the all simulation folder exist when delete_run_path is set to False
    assert os.path.exists(
        os.path.join(simulation_dir, "batch_0/geo_realization_0/simulation_2")
    ), "Simulation folder should be there, something went wrong and was removed!"

    assert os.path.exists(
        os.path.join(simulation_dir, "batch_0/geo_realization_0/simulation_0")
    ), "Simulation folder should be there, something went wrong and was removed"

    assert os.path.exists(
        os.path.join(simulation_dir, "batch_0/geo_realization_0/simulation_1")
    ), "Simulation folder should be there, something went wrong and was removed"


@pytest.mark.integration_test
def test_math_func_auto_scaled_controls(copy_math_func_test_data_to_tmp):
    # Arrange
    config = EverestConfig.load_file("config_minimal.yml")
    config.controls[0].auto_scale = True
    config.controls[0].scaled_range = [0.3, 0.7]

    # Convergence is slower that's why more batches and start closer to final solution?
    config.controls[0].initial_guess = 0.2
    config.optimization.max_batch_num = 10
    config_dict = {
        **config.model_dump(exclude_none=True),
        "input_constraints": [
            {"weights": {"point.x": 1.0, "point.y": 1.0}, "upper_bound": 0.5}
        ],
    }
    config_dict["forward_model"][0] += " --scaling -1 1 0.3 0.7"
    config = EverestConfig.model_validate(config_dict)

    # Act
    run_model = EverestRunModel.create(config)
    evaluator_server_config = EvaluatorServerConfig()
    run_model.run_experiment(evaluator_server_config)

    # Assert
    x, y, z = (run_model.result.controls["point." + p] for p in ("x", "y", "z"))

    assert x == pytest.approx(0.25, abs=0.05)
    assert y == pytest.approx(0.25, abs=0.05)
    assert z == pytest.approx(0.5, abs=0.05)

    # Check optimum value
    optim = -run_model.result.total_objective  # distance is provided as -distance
    expected_dist = 0.25**2 + 0.25**2
    assert expected_dist == pytest.approx(optim, abs=0.05)


@pytest.mark.integration_test
def test_math_func_auto_scaled_objectives(copy_math_func_test_data_to_tmp):
    config = EverestConfig.load_file("config_multiobj.yml")
    config_dict = config.model_dump(exclude_none=True)

    # Normalize only distance_p:
    config_dict["objective_functions"][0]["auto_scale"] = True
    config_dict["objective_functions"][0]["scale"] = 1.0

    # We two batches, the first to do the auto-scaling,
    # the second is the initial function evaluation:
    config_dict["optimization"]["max_batch_num"] = 2

    config = EverestConfig.model_validate(config_dict)
    run_model = EverestRunModel.create(config)
    evaluator_server_config = EvaluatorServerConfig()
    run_model.run_experiment(evaluator_server_config)
    optim = run_model.result.total_objective

    expected_p = 1.0  # normalized
    expected_q = 4.75  # not normalized
    total = -(expected_p * 0.5 + expected_q * 0.25) / (0.5 + 0.25)

    assert total == optim


@pytest.mark.integration_test
def test_math_func_auto_scaled_constraints(copy_math_func_test_data_to_tmp):
    config = EverestConfig.load_file("config_advanced.yml")
    config_dict = config.model_dump(exclude_none=True)

    # control number of batches, no need for full convergence:
    config_dict["optimization"]["convergence_tolerance"] = 1e-10
    config_dict["optimization"]["max_batch_num"] = 3

    # Run with auto_scaling:
    config_dict["environment"]["output_folder"] = "output_auto_scale"
    config_dict["output_constraints"][0]["auto_scale"] = True
    config_dict["output_constraints"][0]["scale"] = 1.0
    config = EverestConfig.model_validate(config_dict)
    run_model = EverestRunModel.create(config)
    evaluator_server_config = EvaluatorServerConfig()
    run_model.run_experiment(evaluator_server_config)
    result1 = run_model.result

    # Run the equivalent without auto-scaling:
    config_dict["environment"]["output_folder"] = "output_manual_scale"
    config_dict["output_constraints"][0]["auto_scale"] = False
    config_dict["output_constraints"][0]["scale"] = 0.25  # x(0)
    # We need one batch less, no auto-scaling:
    config_dict["optimization"]["max_batch_num"] -= 1
    config = EverestConfig.model_validate(config_dict)
    run_model = EverestRunModel.create(config)
    evaluator_server_config = EvaluatorServerConfig()
    run_model.run_experiment(evaluator_server_config)
    result2 = run_model.result

    assert result1.total_objective == pytest.approx(result2.total_objective)
    assert np.allclose(
        np.fromiter(result1.controls.values(), dtype=np.float64),
        np.fromiter(result2.controls.values(), dtype=np.float64),
    )
