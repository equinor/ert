import pytest

from ert.run_models.everest_run_model import EverestRunModel
from everest.config import EverestConfig
from everest.config.sampler_config import SamplerConfig

CONFIG_FILE_ADVANCED = "config_advanced_scipy.yml"


def test_sampler_uniform(
    copy_math_func_test_data_to_tmp, evaluator_server_config_generator
):
    config = EverestConfig.load_file(CONFIG_FILE_ADVANCED)
    config.controls[0].sampler = SamplerConfig(**{"method": "uniform"})

    run_model = EverestRunModel.create(config)
    evaluator_server_config = evaluator_server_config_generator(run_model)
    run_model.run_experiment(evaluator_server_config)

    point_names = ["x-0", "x-1", "x-2"]
    # Check resulting points
    x0, x1, x2 = (run_model.result.controls["point_" + p] for p in point_names)
    assert x0 == pytest.approx(0.1, abs=0.025)
    assert x1 == pytest.approx(0.0, abs=0.025)
    assert x2 == pytest.approx(0.4, abs=0.025)

    # Check optimum value
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


def test_sampler_mixed(
    copy_math_func_test_data_to_tmp, evaluator_server_config_generator
):
    config = EverestConfig.load_file(CONFIG_FILE_ADVANCED)
    config.controls[0].variables[0].sampler = SamplerConfig(**{"method": "uniform"})
    config.controls[0].variables[1].sampler = SamplerConfig(**{"method": "norm"})
    config.controls[0].variables[2].sampler = SamplerConfig(**{"method": "uniform"})

    run_model = EverestRunModel.create(config)
    evaluator_server_config = evaluator_server_config_generator(run_model)
    run_model.run_experiment(evaluator_server_config)

    point_names = ["x-0", "x-1", "x-2"]
    # Check resulting points
    x0, x1, x2 = (run_model.result.controls["point_" + p] for p in point_names)
    assert x0 == pytest.approx(0.1, abs=0.025)
    assert x1 == pytest.approx(0.0, abs=0.025)
    assert x2 == pytest.approx(0.4, abs=0.025)

    # Check optimum value
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
    assert expected_opt == pytest.approx(
        run_model.result.total_objective,
        abs=0.001,
    )
