import pytest

from ert.run_models.everest_run_model import EverestRunModel
from everest.config import EverestConfig

CONFIG_FILE_ADVANCED = "config_advanced.yml"


@pytest.mark.integration_test
def test_fix_control(
    copy_math_func_test_data_to_tmp, evaluator_server_config_generator
):
    config = EverestConfig.load_file(CONFIG_FILE_ADVANCED)
    config.controls[0].variables[0].enabled = False
    config.optimization.max_batch_num = 2

    run_model = EverestRunModel.create(config)
    evaluator_server_config = evaluator_server_config_generator(run_model)
    run_model.run_experiment(evaluator_server_config)

    # Check that the first variable remains fixed:
    assert run_model.result.controls["point_x-0"] == config.controls[0].initial_guess
