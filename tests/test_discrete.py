from everest.config import EverestConfig
from everest.suite import _EverestWorkflow

from tests.utils import relpath, tmpdir

CONFIG_PATH = relpath("..", "examples", "math_func")
CONFIG_DISCRETE = "config_discrete.yml"


@tmpdir(CONFIG_PATH)
def test_discrete_optimizer():
    config = EverestConfig.load_file(CONFIG_DISCRETE)

    workflow = _EverestWorkflow(config)
    assert workflow is not None
    workflow.start_optimization()

    assert workflow.result.controls["point_x"] == 3
    assert workflow.result.controls["point_y"] == 7
