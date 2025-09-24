import pytest

from ert.ensemble_evaluator.config import EvaluatorServerConfig
from ert.run_models.everest_run_model import EverestRunModel
from everest.config import EverestConfig
from tests.everest.utils import get_optimal_result


@pytest.mark.integration_test
@pytest.mark.usefixtures("use_site_configurations_with_no_queue_options")
def test_objective_type(copy_math_func_test_data_to_tmp):
    # Arrange
    config = EverestConfig.load_file("config_minimal.yml")
    config_dict = {
        **config.model_dump(exclude_none=True),
        "objective_functions": [
            {"name": "distance", "weight": 1.0},
            {"name": "stddev", "weight": 1.0, "type": "stddev"},
        ],
        "model": {"realizations": [0, 1]},
        "forward_model": [
            (
                "distance3 --point-file point.json --realization <GEO_ID> "
                "--target 0.5 0.5 0.5 --out distance"
            ),
            (
                "distance3 --point-file point.json --realization <GEO_ID> "
                "--target 0.5 0.5 0.5 --out stddev"
            ),
        ],
        "simulator": {"queue_system": {"name": "local", "max_running": 2}},
    }
    config = EverestConfig.model_validate(config_dict)
    # Act
    run_model = EverestRunModel.create(config)
    evaluator_server_config = EvaluatorServerConfig()
    run_model.run_experiment(evaluator_server_config)

    optimal_result = get_optimal_result(config.optimization_output_dir)

    # Assert
    x0, x1, x2 = (optimal_result.controls["point." + p] for p in ["x", "y", "z"])
    assert x0 == pytest.approx(0.5, abs=0.025)
    assert x1 == pytest.approx(0.5, abs=0.025)
    assert x2 == pytest.approx(0.5, abs=0.025)

    assert optimal_result.total_objective < 0.0
