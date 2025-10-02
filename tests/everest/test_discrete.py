import pytest

from ert.ensemble_evaluator.config import EvaluatorServerConfig
from ert.plugins import ErtPluginContext
from ert.run_models.everest_run_model import EverestRunModel
from everest.config import EverestConfig
from tests.everest.utils import get_optimal_result


@pytest.mark.integration_test
@pytest.mark.usefixtures("use_site_configurations_with_no_queue_options")
def test_discrete_optimizer(copy_math_func_test_data_to_tmp):
    # Arrange
    config = EverestConfig.load_file("config_minimal.yml")
    config_dict = {
        **config.model_dump(exclude_none=True),
        "optimization": {
            "algorithm": "scipy/differential_evolution",
            "max_function_evaluations": 4,
            "parallel": False,
            "backend_options": {"rng": 4},
        },
        "controls": [
            {
                "name": "point",
                "type": "generic_control",
                "min": 0,
                "max": 10,
                "control_type": "integer",
                "initial_guess": 0,
                "perturbation_magnitude": 0.01,
                "variables": [{"name": "x"}, {"name": "y"}],
            }
        ],
        "input_constraints": [
            {"weights": {"point.x": 1.0, "point.y": 1.0}, "upper_bound": 10}
        ],
        "install_jobs": [{"name": "discrete", "executable": "jobs/discrete.py"}],
        "forward_model": ["discrete --point-file point.json --out distance"],
    }
    config = EverestConfig.model_validate(config_dict)
    # Act
    with ErtPluginContext() as runtime_plugins:
        run_model = EverestRunModel.create(config, runtime_plugins=runtime_plugins)

    evaluator_server_config = EvaluatorServerConfig()
    run_model.run_experiment(evaluator_server_config)

    optimal_result = get_optimal_result(config.optimization_output_dir)

    # Assert
    assert optimal_result.controls["point.x"] == 3
    assert optimal_result.controls["point.y"] == 7
