import pytest

from ert.ensemble_evaluator.config import EvaluatorServerConfig
from ert.run_models.everest_run_model import EverestRunModel
from everest.config import EverestConfig


@pytest.mark.integration_test
def test_discrete_optimizer(copy_math_func_test_data_to_tmp):
    # Arrange
    config = EverestConfig.load_file("config_minimal.yml")
    config_dict = {
        **config.model_dump(exclude_none=True),
        "optimization": {
            "backend": "scipy",
            "algorithm": "differential_evolution",
            "max_function_evaluations": 4,
            "parallel": False,
            "backend_options": {"seed": 9},
        },
        "controls": [
            {
                "name": "point",
                "type": "generic_control",
                "min": 0,
                "max": 10,
                "control_type": "integer",
                "initial_guess": 0,
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
    run_model = EverestRunModel.create(config)
    evaluator_server_config = EvaluatorServerConfig()
    run_model.run_experiment(evaluator_server_config)

    # Assert
    assert run_model.result.controls["point.x"] == 3
    assert run_model.result.controls["point.y"] == 7
