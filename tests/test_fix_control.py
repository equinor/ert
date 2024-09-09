from everest.config import EverestConfig
from everest.suite import _EverestWorkflow

from tests.utils import relpath, tmpdir

CONFIG_PATH = relpath("..", "examples", "math_func")
CONFIG_FILE_ADVANCED = "config_advanced_scipy.yml"


@tmpdir(CONFIG_PATH)
def test_fix_control():
    config = EverestConfig.load_file(CONFIG_FILE_ADVANCED)
    config.controls[0].variables[0].enabled = False

    workflow = _EverestWorkflow(config)
    assert workflow is not None
    workflow.start_optimization()

    # Check that the first variable remains fixed:
    assert workflow.result.controls["point_x-0"] == config.controls[0].initial_guess
