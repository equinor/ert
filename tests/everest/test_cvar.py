import pytest

from ert.run_models.everest_run_model import EverestRunModel
from everest.config import EverestConfig

CONFIG_FILE_CVAR = "config_cvar.yml"


def test_mathfunc_cvar(
    copy_math_func_test_data_to_tmp, evaluator_server_config_generator
):
    config = EverestConfig.load_file(CONFIG_FILE_CVAR)

    run_model = EverestRunModel.create(config)
    evaluator_server_config = evaluator_server_config_generator(run_model)
    run_model.run_experiment(evaluator_server_config)

    # Check resulting points
    x0, x1, x2 = (run_model.result.controls["point_" + p] for p in ["x", "y", "z"])

    assert x0 == pytest.approx(0.5, 0.05)
    assert x1 == pytest.approx(0.5, 0.05)
    assert x2 == pytest.approx(0.5, 0.05)

    total_objective = run_model.result.total_objective
    assert total_objective <= 0.001
    assert total_objective >= -0.001
