from everest.config import EverestConfig
from everest.suite import _EverestWorkflow

CONFIG_DISCRETE = "config_discrete.yml"


def test_discrete_optimizer(copy_math_func_test_data_to_tmp):
    config = EverestConfig.load_file(CONFIG_DISCRETE)

    workflow = _EverestWorkflow(config)
    assert workflow is not None
    workflow.start_optimization()

    assert workflow.result.controls["point_x"] == 3
    assert workflow.result.controls["point_y"] == 7
