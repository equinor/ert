import pytest

from ert.run_models.everest_run_model import EverestRunModel
from everest.config import (
    ControlConfig,
    EverestConfig,
    InputConstraintConfig,
    InstallJobConfig,
    OptimizationConfig,
)


@pytest.mark.integration_test
def test_discrete_optimizer(
    copy_math_func_test_data_to_tmp, evaluator_server_config_generator
):
    # Arrange
    config = EverestConfig.load_file("config_minimal.yml")
    config.controls = [
        ControlConfig(
            name="point",
            type="generic_control",
            min=0,
            max=10,
            control_type="integer",
            initial_guess=0,
            variables=[{"name": "x"}, {"name": "y"}],
        )
    ]
    config.input_constraints = [
        InputConstraintConfig(weights={"point.x": 1.0, "point.y": 1.0}, upper_bound=10)
    ]
    config.optimization = OptimizationConfig(
        backend="scipy",
        algorithm="differential_evolution",
        max_function_evaluations=4,
        parallel=False,
        backend_options={"seed": 9},
    )
    config.install_jobs = [InstallJobConfig(name="discrete", source="jobs/DISCRETE")]
    config.forward_model = ["discrete --point-file point.json --out distance"]

    # Act
    run_model = EverestRunModel.create(config)
    evaluator_server_config = evaluator_server_config_generator(run_model)
    run_model.run_experiment(evaluator_server_config)

    # Assert
    assert run_model.result.controls["point_x"] == 3
    assert run_model.result.controls["point_y"] == 7
