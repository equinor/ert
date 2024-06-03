import numpy as np
from everest.config import EverestConfig
from everest.simulator import Simulator
from everest.suite import _EverestWorkflow

from tests.utils import relpath, tmp

CONFIG_PATH = relpath("..", "examples", "math_func")
CONFIG_FILE_RESTART = "config_restart.yml"


def test_cache_optimizer(monkeypatch):
    cached = False
    original_call = Simulator.__call__

    def new_call(*args):
        nonlocal cached
        result = original_call(*args)
        # Without caching there should be 10 evaluations:
        if (result.evaluation_ids >= 0).sum() < 10:
            cached = True
        return result

    monkeypatch.setattr(Simulator, "__call__", new_call)

    with tmp(CONFIG_PATH):
        config = EverestConfig.load_file(CONFIG_FILE_RESTART)
        config.simulator.enable_cache = False

        workflow = _EverestWorkflow(config)
        assert workflow is not None
        workflow.start_optimization()

        x1 = np.fromiter(
            (workflow.result.controls["point_0_" + p] for p in ["x-0", "x-1", "x-2"]),
            dtype=np.float64,
        )

    assert not cached
    assert np.allclose(x1, [0.1, 0.0, 0.4], atol=0.02)

    with tmp(CONFIG_PATH):
        config = EverestConfig.load_file(CONFIG_FILE_RESTART)
        config.simulator.enable_cache = True

        workflow = _EverestWorkflow(config)
        assert workflow is not None
        workflow.start_optimization()

        x2 = np.fromiter(
            (workflow.result.controls["point_0_" + p] for p in ["x-0", "x-1", "x-2"]),
            dtype=np.float64,
        )

    assert cached
    assert np.allclose(x1, x2, atol=np.finfo(np.float64).eps)
