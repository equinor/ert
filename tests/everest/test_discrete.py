import pytest

from ert.run_models.everest_run_model import EverestRunModel
from everest.config import EverestConfig
from everest.config.input_constraint_config import InputConstraintConfig


@pytest.mark.integration_test
def test_discrete_optimizer(
    copy_math_func_test_data_to_tmp, evaluator_server_config_generator
):
    # Arrange
    config = EverestConfig.load_file("config_minimal.yml")
    config.controls[0].min = 0
    config.controls[0].max = 10
    config.controls[0].control_type = "integer"
    config.controls[0].initial_guess = 0
    config.controls[0].variables.pop()
    config.input_constraints = [
        InputConstraintConfig(weights={"point.x": 1.0, "point.y": 1.0}, upper_bound=10)
    ]
    config.optimization.backend = "scipy"
    config.optimization.algorithm = "differential_evolution"
    config.optimization.max_function_evaluations = 4
    config.optimization.parallel = False
    config.optimization.backend_options = {"seed": 9}
    config.install_jobs[0].name = "discrete"
    config.install_jobs[0].source = "jobs/DISCRETE"
    config.forward_model[0] = "discrete --point-file point.json --out distance"

    # Act
    run_model = EverestRunModel.create(config)
    evaluator_server_config = evaluator_server_config_generator(run_model)
    run_model.run_experiment(evaluator_server_config)

    # Assert
    assert run_model.result.controls["point_x"] == 3
    assert run_model.result.controls["point_y"] == 7
