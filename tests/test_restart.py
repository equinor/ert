import pytest
from everest.config import EverestConfig
from everest.suite import _EverestWorkflow

from tests.utils import relpath, tmpdir

CONFIG_PATH = relpath("..", "examples", "math_func")
CONFIG_FILE_RESTART = "config_restart.yml"


@tmpdir(CONFIG_PATH)
def test_restart_optimizer():
    config = EverestConfig.load_file(CONFIG_FILE_RESTART)

    workflow = _EverestWorkflow(config)
    assert workflow is not None
    workflow.start_optimization()

    point_names = ["x-0", "x-1", "x-2"]
    # Check resulting points
    x0, x1, x2 = (workflow.result.controls["point_0_" + p] for p in point_names)
    assert x0 == pytest.approx(0.1, abs=0.025)
    assert x1 == pytest.approx(0.0, abs=0.025)
    assert x2 == pytest.approx(0.4, abs=0.025)

    # Since we restarted once, we have twice the number of
    # max_function_evaluations:
    assert workflow.result.batch == 5
