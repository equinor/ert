import numpy

from everest.config import EverestConfig
from everest.suite import _EverestWorkflow
from tests.utils import relpath, tmpdir

CONFIG_PATH = relpath("..", "examples", "math_func")
CONFIG_FILE_EVALUATION = "config_ensemble_evaluation.yml"

# This test uses an experimental feature to use an ensemble evaluation step to
# filter faulty realizations that would otherwise make the optimization fail.
# See the config file for more details.


@tmpdir(CONFIG_PATH)
def test_ensemble_evaluation():
    config = EverestConfig.load_file(CONFIG_FILE_EVALUATION)

    workflow = _EverestWorkflow(config)
    assert workflow is not None
    workflow.start_optimization()

    # Check resulting points
    x0, x1, x2 = (workflow.result.controls[f"point_0_{c}"] for c in ("x", "y", "z"))

    assert numpy.allclose([x0, x1, x2], [0.5, 0.5, 0.5], atol=0.02)
