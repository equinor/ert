from everest.config import EverestConfig
from everest.suite import _EverestWorkflow

CONFIG_FILE_ADVANCED = "config_advanced_scipy.yml"


def test_fix_control(copy_math_func_test_data_to_tmp):
    config = EverestConfig.load_file(CONFIG_FILE_ADVANCED)
    config.controls[0].variables[0].enabled = False

    workflow = _EverestWorkflow(config)
    assert workflow is not None
    workflow.start_optimization()

    # Check that the first variable remains fixed:
    assert workflow.result.controls["point_x-0"] == config.controls[0].initial_guess
