import numpy as np
import pytest

from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.run_models.everest_run_model import EverestRunModel
from everest.config import EverestConfig


@pytest.mark.integration_test
def test_simulator_cache(copy_math_func_test_data_to_tmp):
    n_invocations = 0

    def new_call(*args):
        nonlocal n_invocations
        result = original_call(*args)
        n_invocations += 1
        return result

    config = EverestConfig.load_file("config_minimal.yml")
    config_dict = config.model_dump(exclude_none=True)
    config = EverestConfig.model_validate(config_dict)

    run_model = EverestRunModel.create(config)
    evaluator_server_config = EvaluatorServerConfig()

    # Modify the forward model function to track number of calls:
    original_call = run_model._evaluate_and_postprocess
    run_model._evaluate_and_postprocess = new_call

    # First run populates the cache:
    run_model.run_experiment(evaluator_server_config)
    assert n_invocations > 0
    variables1 = list(run_model.result.controls.values())
    assert np.allclose(variables1, [0.5, 0.5, 0.5], atol=0.02)

    # Now do another run, where the functions should come from the cache:
    n_invocations = 0

    # The batch_id was used as a stopping criterion, so it must be reset:
    run_model._batch_id = 0

    run_model.run_experiment(evaluator_server_config)
    assert n_invocations == 0
    variables2 = list(run_model.result.controls.values())
    assert np.array_equal(variables1, variables2)
