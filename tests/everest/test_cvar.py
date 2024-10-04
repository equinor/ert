import pytest

from everest.config import EverestConfig
from everest.suite import _EverestWorkflow

CONFIG_FILE_CVAR = "config_cvar.yml"


def test_mathfunc_cvar(copy_math_func_test_data_to_tmp):
    config = EverestConfig.load_file(CONFIG_FILE_CVAR)

    workflow = _EverestWorkflow(config)
    assert workflow is not None
    workflow.start_optimization()

    # Check resulting points
    x0, x1, x2 = (workflow.result.controls["point_" + p] for p in ["x", "y", "z"])

    assert x0 == pytest.approx(0.5, 0.05)
    assert x1 == pytest.approx(0.5, 0.05)
    assert x2 == pytest.approx(0.5, 0.05)

    total_objective = workflow.result.total_objective
    assert total_objective <= 0.001
    assert total_objective >= -0.001
