import pytest

from ert.run_models.everest_run_model import EverestRunModel
from everest.config import EverestConfig
from everest.config.cvar_config import CVaRConfig


@pytest.mark.integration_test
def test_mathfunc_cvar(
    copy_math_func_test_data_to_tmp, evaluator_server_config_generator
):
    # Arrange
    config = EverestConfig.load_file("config_minimal.yml")
    config.controls[0].perturbation_magnitude = 0.01
    config.controls[0].min = -2.0
    config.controls[0].max = 2.0
    config.controls[0].initial_guess = 0.0
    config.optimization.backend = "scipy"
    config.optimization.algorithm = "slsqp"
    config.optimization.cvar = CVaRConfig(percentile=0.5)
    config.optimization.max_batch_num = 5
    config.model.realizations = [0, 1]
    config.forward_model[0] = (
        "distance3 --point-file point.json --realization <GEO_ID> --target 0.5 0.5 0.5 --out distance"
    )

    # Act
    run_model = EverestRunModel.create(config)
    evaluator_server_config = evaluator_server_config_generator(run_model)
    run_model.run_experiment(evaluator_server_config)

    # Assert
    x0, x1, x2 = (run_model.result.controls["point_" + p] for p in ["x", "y", "z"])

    assert x0 == pytest.approx(0.5, 0.05)
    assert x1 == pytest.approx(0.5, 0.05)
    assert x2 == pytest.approx(0.5, 0.05)

    total_objective = run_model.result.total_objective
    assert total_objective <= 0.001
    assert total_objective >= -0.001
