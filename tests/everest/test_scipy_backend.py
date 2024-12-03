import pytest

from ert.run_models.everest_run_model import EverestRunModel
from everest.config import EverestConfig


def test_scipy_backend(
    copy_math_func_test_data_to_tmp, evaluator_server_config_generator
):
    # Arrange
    config = EverestConfig.load_file("config_advanced.yml")
    config.optimization.backend = "scipy"
    config.optimization.algorithm = "SLSQP"
    config.optimization.convergence_tolerance = 0.001
    config.optimization.constraint_tolerance = 0.001
    config.optimization.perturbation_num = 7
    config.optimization.speculative = True
    config.optimization.max_batch_num = 4
    config.optimization.backend_options = {"maxiter": 100}

    # Act
    run_model = EverestRunModel.create(config)
    evaluator_server_config = evaluator_server_config_generator(run_model)
    run_model.run_experiment(evaluator_server_config)

    # Assert
    point_names = ["x-0", "x-1", "x-2"]
    x0, x1, x2 = (run_model.result.controls["point_" + p] for p in point_names)
    assert x0 == pytest.approx(0.1, abs=0.025)
    assert x1 == pytest.approx(0.0, abs=0.025)
    assert x2 == pytest.approx(0.4, abs=0.025)

    # Optimal value
    assert pytest.approx(run_model.result.total_objective, abs=0.01) == -(
        0.25 * (1.6**2 + 1.5**2 + 0.1**2) + 0.75 * (0.4**2 + 0.5**2 + 0.1**2)
    )

    # Expected distance is the weighted average of the (squared) distances
    #  from (x, y, z) to (-1.5, -1.5, 0.5) and (0.5, 0.5, 0.5)
    w = config.model.realizations_weights
    assert w == [0.25, 0.75]
    dist_0 = (x0 + 1.5) ** 2 + (x1 + 1.5) ** 2 + (x2 - 0.5) ** 2
    dist_1 = (x0 - 0.5) ** 2 + (x1 - 0.5) ** 2 + (x2 - 0.5) ** 2
    expected_opt = -(w[0] * (dist_0) + w[1] * (dist_1))
    assert expected_opt == pytest.approx(run_model.result.total_objective, abs=0.001)
