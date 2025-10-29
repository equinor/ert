import pytest

from ert.base_model_context import use_runtime_plugins
from ert.ensemble_evaluator.config import EvaluatorServerConfig
from ert.plugins import get_site_plugins
from ert.run_models.everest_run_model import EverestRunModel
from everest.config import EverestConfig
from tests.everest.utils import get_optimal_result


@pytest.mark.integration_test
@pytest.mark.usefixtures("use_site_configurations_with_no_queue_options")
def test_mathfunc_cvar(copy_math_func_test_data_to_tmp):
    config = EverestConfig.load_file("config_minimal.yml")
    config_dict = {
        **config.model_dump(exclude_none=True),
        "optimization": {
            "algorithm": "scipy/slsqp",
            "cvar": {"percentile": 0.5},
            "max_batch_num": 5,
        },
        "model": {"realizations": [0, 1]},
        "forward_model": [
            (
                "distance3 --point-file point.json --realization <GEO_ID> "
                "--target 0.5 0.5 0.5 --out distance"
            )
        ],
    }
    config = EverestConfig.model_validate(config_dict)
    # Act
    site_plugins = get_site_plugins()
    with use_runtime_plugins(site_plugins):
        run_model = EverestRunModel.create(config, runtime_plugins=site_plugins)

    evaluator_server_config = EvaluatorServerConfig()
    run_model.run_experiment(evaluator_server_config)

    optimal_result = get_optimal_result(config.optimization_output_dir)

    # Assert
    x0, x1, x2 = (optimal_result.controls["point." + p] for p in ["x", "y", "z"])

    assert x0 == pytest.approx(0.5, 0.05)
    assert x1 == pytest.approx(0.5, 0.05)
    assert x2 == pytest.approx(0.5, 0.05)

    total_objective = optimal_result.total_objective
    assert total_objective <= 0.001
    assert total_objective >= -0.001
