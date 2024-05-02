import pytest

from everest.config import EverestConfig
from everest.suite import _EverestWorkflow
from tests.utils import relpath, tmpdir

CONFIG_PATH = relpath("..", "examples", "math_func")
CONFIG_FILE_STDDEV = "config_stddev.yml"


@tmpdir(CONFIG_PATH)
def test_mathfunc_stddev():
    config = EverestConfig.load_file(CONFIG_FILE_STDDEV)

    workflow = _EverestWorkflow(config)
    assert workflow is not None
    workflow.start_optimization()

    # Check resulting points
    x0, x1, x2 = (workflow.result.controls["point_0_" + p] for p in ["x", "y", "z"])
    assert x0 == pytest.approx(0.5, abs=0.025)
    assert x1 == pytest.approx(0.5, abs=0.025)
    assert x2 == pytest.approx(0.5, abs=0.025)

    assert workflow.result.total_objective < 0.0
