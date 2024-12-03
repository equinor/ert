import pytest

from ert.run_models.everest_run_model import EverestRunModel
from everest.config import (
    EverestConfig,
    ModelConfig,
    ObjectiveFunctionConfig,
)


@pytest.mark.integration_test
def test_objective_type(
    copy_math_func_test_data_to_tmp, evaluator_server_config_generator
):
    # Arrange
    config = EverestConfig.load_file("config_minimal.yml")
    config.objective_functions = [
        ObjectiveFunctionConfig(name="distance", weight=1.0),
        ObjectiveFunctionConfig(
            name="stddev", weight=1.0, type="stddev", alias="distance"
        ),
    ]
    config.model = ModelConfig(realizations=[0, 1])
    config.forward_model = [
        "distance3 --point-file point.json --realization <GEO_ID> --target 0.5 0.5 0.5 --out distance"
    ]

    # Act
    run_model = EverestRunModel.create(config)
    evaluator_server_config = evaluator_server_config_generator(run_model)
    run_model.run_experiment(evaluator_server_config)

    # Assert
    x0, x1, x2 = (run_model.result.controls["point_" + p] for p in ["x", "y", "z"])
    assert x0 == pytest.approx(0.5, abs=0.025)
    assert x1 == pytest.approx(0.5, abs=0.025)
    assert x2 == pytest.approx(0.5, abs=0.025)

    assert run_model.result.total_objective < 0.0
