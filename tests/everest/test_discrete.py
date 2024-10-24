from ert.run_models.everest_run_model import EverestRunModel
from everest.config import EverestConfig

CONFIG_DISCRETE = "config_discrete.yml"


def test_discrete_optimizer(
    copy_math_func_test_data_to_tmp, evaluator_server_config_generator
):
    config = EverestConfig.load_file(CONFIG_DISCRETE)

    run_model = EverestRunModel.create(config)
    evaluator_server_config = evaluator_server_config_generator(run_model)
    run_model.run_experiment(evaluator_server_config)

    assert run_model.result.controls["point_x"] == 3
    assert run_model.result.controls["point_y"] == 7
