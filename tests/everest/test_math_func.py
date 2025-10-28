import os
from pathlib import Path

import pytest
import yaml

from ert.base_model_context import use_runtime_plugins
from ert.ensemble_evaluator.config import EvaluatorServerConfig
from ert.plugins import ErtPluginContext
from ert.run_models.everest_run_model import EverestRunModel
from ert.storage import open_storage
from everest.config import EverestConfig
from everest.util import makedirs_if_needed
from tests.everest.utils import get_optimal_result

CONFIG_FILE_MULTIOBJ = "config_multiobj.yml"
CONFIG_FILE_ADVANCED = "config_advanced.yml"


@pytest.mark.xdist_group("math_func/config_multiobj.yml")
@pytest.mark.integration_test
def test_math_func_multiobj(cached_example):
    config_path, config_file, _, _ = cached_example("math_func/config_multiobj.yml")

    config = EverestConfig.load_file(Path(config_path) / config_file)

    result = get_optimal_result(config.optimization_output_dir)

    # Check resulting points
    x, y, z = (result.controls["point." + p] for p in ("x", "y", "z"))
    assert x == pytest.approx(0.0, abs=0.05)
    assert y == pytest.approx(0.0, abs=0.05)
    assert z == pytest.approx(0.5, abs=0.05)

    # The overall optimum is a weighted average of the objectives
    expected = -(0.5 * 2 * 0.5**2 + 0.25 * 2 * 1.5**2) / 0.75
    assert result.total_objective == pytest.approx(expected, abs=0.01)


@pytest.mark.xdist_group("math_func/config_advanced.yml")
@pytest.mark.integration_test
def test_math_func_advanced(cached_example):
    config_path, config_file, _, _ = cached_example("math_func/config_advanced.yml")

    config = EverestConfig.load_file(Path(config_path) / config_file)
    result = get_optimal_result(config.optimization_output_dir)

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
@pytest.mark.usefixtures("use_site_configurations_with_no_queue_options")
def test_remove_run_path(copy_math_func_test_data_to_tmp):
    with open("config_minimal.yml", encoding="utf-8") as file:
        config_yaml = yaml.safe_load(file)
        config_yaml["simulator"] = {"delete_run_path": True}
        config_yaml["install_jobs"].append(
            {"name": "toggle_failure", "executable": "jobs/fail_simulation.py"}
        )
        config_yaml["forward_model"].append("toggle_failure --fail perturbation_1")
    with open("config.yml", "w", encoding="utf-8") as fout:
        yaml.dump(config_yaml, fout)
    config = EverestConfig.load_file("config.yml")

    simulation_dir = config.simulation_dir

    site_plugins = ErtPluginContext.get_site_plugins()
    with use_runtime_plugins(site_plugins):
        run_model = EverestRunModel.create(config, runtime_plugins=site_plugins)

    evaluator_server_config = EvaluatorServerConfig()
    run_model.run_experiment(evaluator_server_config)

    # Check the failed simulation folder still exists
    assert os.path.exists(
        os.path.join(simulation_dir, "batch_0/realization_0/perturbation_1")
    ), "Simulation folder should be there, something went wrong and was removed!"

    # Check the successful simulation folders do not exist
    assert not os.path.exists(
        os.path.join(simulation_dir, "batch_0/realization_0/evaluation_0")
    ), "Simulation folder should not be there, something went wrong!"

    assert not os.path.exists(
        os.path.join(simulation_dir, "batch_0/realization_0/simulation_1")
    ), "Simulation folder should not be there, something went wrong!"

    # Manually rolling the output folder between two runs
    makedirs_if_needed(config.output_dir, roll_if_exists=True)

    config.simulator.delete_run_path = False
    site_plugins = ErtPluginContext.get_site_plugins()
    with use_runtime_plugins(site_plugins):
        run_model = EverestRunModel.create(config, runtime_plugins=site_plugins)

    evaluator_server_config = EvaluatorServerConfig()
    run_model.run_experiment(evaluator_server_config)

    # Check the all simulation folder exist when delete_run_path is set to False
    assert os.path.exists(
        os.path.join(simulation_dir, "batch_0/realization_0/perturbation_1")
    ), "Simulation folder should be there, something went wrong and was removed!"

    assert os.path.exists(
        os.path.join(simulation_dir, "batch_0/realization_0/evaluation_0")
    ), "Simulation folder should be there, something went wrong and was removed"

    assert os.path.exists(
        os.path.join(simulation_dir, "batch_0/realization_0/perturbation_0")
    ), "Simulation folder should be there, something went wrong and was removed"


@pytest.mark.integration_test
@pytest.mark.usefixtures("use_site_configurations_with_no_queue_options")
def test_math_func_auto_scaled_controls(copy_math_func_test_data_to_tmp):
    # Arrange
    config = EverestConfig.load_file("config_minimal.yml")
    config.controls[0].scaled_range = (0.3, 0.7)

    # Convergence is slower that's why more batches and start closer to final solution?
    config.controls[0].initial_guess = 0.2
    config.optimization.max_batch_num = 8
    config_dict = {
        **config.model_dump(exclude_none=True),
        "input_constraints": [
            {
                "weights": {"point.x": 1.0, "point.y": 1.0},
                "lower_bound": 0.2,
                "upper_bound": 0.5,
            }
        ],
        "simulator": {"queue_system": {"name": "local", "max_running": 2}},
    }
    config = EverestConfig.model_validate(config_dict)

    # Act
    site_plugins = ErtPluginContext.get_site_plugins()
    with use_runtime_plugins(site_plugins):
        run_model = EverestRunModel.create(config, runtime_plugins=site_plugins)
    evaluator_server_config = EvaluatorServerConfig()
    run_model.run_experiment(evaluator_server_config)

    optimal_result = get_optimal_result(config.optimization_output_dir)

    # Assert
    x, y, z = (optimal_result.controls["point." + p] for p in ("x", "y", "z"))

    assert x == pytest.approx(0.25, abs=0.05)
    assert y == pytest.approx(0.25, abs=0.05)
    assert z == pytest.approx(0.5, abs=0.05)

    # Check optimum value
    optim = -optimal_result.total_objective  # distance is provided as -distance
    expected_dist = 0.25**2 + 0.25**2
    assert expected_dist == pytest.approx(optim, abs=0.05)


@pytest.mark.integration_test
@pytest.mark.usefixtures("use_site_configurations_with_no_queue_options")
def test_math_func_auto_scaled_objectives(copy_math_func_test_data_to_tmp):
    config = EverestConfig.load_file("config_multiobj.yml")
    config_dict = config.model_dump(exclude_none=True)

    config_dict["simulator"] = {"queue_system": {"name": "local", "max_running": 2}}
    del config_dict["objective_functions"][0]["scale"]

    config_dict["environment"]["output_folder"] = "output_no_auto_scale"
    config = EverestConfig.model_validate(config_dict)
    site_plugins = ErtPluginContext.get_site_plugins()
    with use_runtime_plugins(site_plugins):
        run_model = EverestRunModel.create(config, runtime_plugins=site_plugins)
    evaluator_server_config = EvaluatorServerConfig()
    run_model.run_experiment(evaluator_server_config)
    optim1 = get_optimal_result(config.optimization_output_dir).total_objective

    config_dict["environment"]["output_folder"] = "output_auto_scale"
    config_dict["optimization"]["auto_scale"] = True
    config = EverestConfig.model_validate(config_dict)
    site_plugins = ErtPluginContext.get_site_plugins()
    with use_runtime_plugins(site_plugins):
        run_model = EverestRunModel.create(config, runtime_plugins=site_plugins)
    evaluator_server_config = EvaluatorServerConfig()
    run_model.run_experiment(evaluator_server_config)
    optim2 = get_optimal_result(config.optimization_output_dir).total_objective

    assert optim1 != optim2
    assert optim1 == pytest.approx(optim2, abs=0.001)


@pytest.mark.integration_test
@pytest.mark.usefixtures("use_site_configurations_with_no_queue_options")
def test_math_func_auto_scaled_constraints(copy_math_func_test_data_to_tmp):
    config = EverestConfig.load_file("config_advanced.yml")
    config_dict = config.model_dump(exclude_none=True)

    config_dict["simulator"] = {"queue_system": {"name": "local", "max_running": 2}}
    del config_dict["output_constraints"][0]["scale"]

    config_dict["environment"]["output_folder"] = "output_no_auto_scale"
    config = EverestConfig.model_validate(config_dict)
    site_plugins = ErtPluginContext.get_site_plugins()
    with use_runtime_plugins(site_plugins):
        run_model = EverestRunModel.create(config, runtime_plugins=site_plugins)
    evaluator_server_config = EvaluatorServerConfig()
    run_model.run_experiment(evaluator_server_config)
    optim1 = get_optimal_result(config.optimization_output_dir).total_objective

    config_dict["environment"]["output_folder"] = "output_auto_scale"
    config_dict["optimization"]["auto_scale"] = True
    config = EverestConfig.model_validate(config_dict)
    site_plugins = ErtPluginContext.get_site_plugins()
    with use_runtime_plugins(site_plugins):
        run_model = EverestRunModel.create(config, runtime_plugins=site_plugins)
    evaluator_server_config = EvaluatorServerConfig()
    run_model.run_experiment(evaluator_server_config)
    optim2 = get_optimal_result(config.optimization_output_dir).total_objective

    assert optim1 != optim2
    assert optim1 == pytest.approx(optim2, abs=0.01)


@pytest.mark.xdist_group("math_func/config_advanced.yml")
@pytest.mark.integration_test
def test_ensemble_creation(cached_example):
    cached_example("math_func/config_advanced.yml")
    with open_storage("everest_output/simulation_results", "r") as storage:
        ensembles = storage.ensembles
        assert sorted(ensemble.iteration for ensemble in ensembles) == sorted(range(5))


@pytest.mark.integration_test
@pytest.mark.usefixtures("use_site_configurations_with_no_queue_options")
def test_that_math_func_violating_output_constraints_has_no_result(
    copy_math_func_test_data_to_tmp,
):
    config = EverestConfig.load_file("config_advanced.yml")
    config_dict = config.model_dump(exclude_none=True)

    config_dict["simulator"] = {"queue_system": {"name": "local", "max_running": 2}}

    # The first batch violates the output constraint:
    config_dict["optimization"]["max_batch_num"] = 1
    config_dict["controls"][0]["initial_guess"] = 0.05

    config = EverestConfig.model_validate(config_dict)
    site_plugins = ErtPluginContext.get_site_plugins()
    with use_runtime_plugins(site_plugins):
        run_model = EverestRunModel.create(config, runtime_plugins=site_plugins)
    evaluator_server_config = EvaluatorServerConfig()
    run_model.run_experiment(evaluator_server_config)
    optimal_result = get_optimal_result(config.optimization_output_dir)
    assert optimal_result is None  # No feasible result


@pytest.mark.integration_test
@pytest.mark.usefixtures("use_site_configurations_with_no_queue_options")
def test_that_math_func_violating_output_constraints_has_a_result(
    copy_math_func_test_data_to_tmp,
):
    config = EverestConfig.load_file("config_advanced.yml")
    config_dict = config.model_dump(exclude_none=True)

    config_dict["simulator"] = {"queue_system": {"name": "local", "max_running": 2}}

    # The second batch does not violate the output constraint:
    config_dict["optimization"]["max_batch_num"] = 2
    config_dict["controls"][0]["initial_guess"] = 0.05

    config = EverestConfig.model_validate(config_dict)
    site_plugins = ErtPluginContext.get_site_plugins()
    with use_runtime_plugins(site_plugins):
        run_model = EverestRunModel.create(config, runtime_plugins=site_plugins)
    evaluator_server_config = EvaluatorServerConfig()
    run_model.run_experiment(evaluator_server_config)
    optimal_result = get_optimal_result(config.optimization_output_dir)
    assert optimal_result is not None  # Feasible result


@pytest.mark.integration_test
@pytest.mark.usefixtures("use_site_configurations_with_no_queue_options")
def test_that_math_func_violating_input_constraints_has_no_result(
    copy_math_func_test_data_to_tmp,
):
    config = EverestConfig.load_file("config_advanced.yml")
    config_dict = config.model_dump(exclude_none=True)

    config_dict["simulator"] = {"queue_system": {"name": "local", "max_running": 2}}

    # The first batch violates the input constraint:
    config_dict["optimization"]["max_batch_num"] = 1
    config_dict["controls"][0]["initial_guess"] = 0.5

    config = EverestConfig.model_validate(config_dict)
    site_plugins = ErtPluginContext.get_site_plugins()
    with use_runtime_plugins(site_plugins):
        run_model = EverestRunModel.create(config, runtime_plugins=site_plugins)
    evaluator_server_config = EvaluatorServerConfig()
    run_model.run_experiment(evaluator_server_config)
    optimal_result = get_optimal_result(config.optimization_output_dir)
    assert optimal_result is None  # No feasible result


@pytest.mark.integration_test
@pytest.mark.usefixtures("use_site_configurations_with_no_queue_options")
def test_that_math_func_violating_input_constraints_has_a_result(
    copy_math_func_test_data_to_tmp,
):
    config = EverestConfig.load_file("config_advanced.yml")
    config_dict = config.model_dump(exclude_none=True)

    config_dict["simulator"] = {"queue_system": {"name": "local", "max_running": 2}}

    # The second batch does not violate the input constraint:
    config_dict["optimization"]["max_batch_num"] = 2
    config_dict["controls"][0]["initial_guess"] = 0.5

    config = EverestConfig.model_validate(config_dict)
    site_plugins = ErtPluginContext.get_site_plugins()
    with use_runtime_plugins(site_plugins):
        run_model = EverestRunModel.create(config, runtime_plugins=site_plugins)
    evaluator_server_config = EvaluatorServerConfig()
    run_model.run_experiment(evaluator_server_config)
    optimal_result = get_optimal_result(config.optimization_output_dir)
    assert optimal_result is not None  # Feasible result
