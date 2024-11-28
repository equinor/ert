from pathlib import Path

import numpy as np

from ert.config import QueueSystem
from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.run_models.everest_run_model import EverestRunModel
from everest.config import EverestConfig, SimulatorConfig

CONFIG_FILE = "config_advanced_scipy.yml"


def test_simulator_cache(copy_math_func_test_data_to_tmp):
    n_evals = 0

    def new_call(*args):
        nonlocal n_evals
        result = original_call(*args)
        n_evals += (result.evaluation_ids >= 0).sum()
        return result

    config = EverestConfig.load_file(CONFIG_FILE)
    config.simulator = SimulatorConfig(enable_cache=True)

    run_model = EverestRunModel.create(config)

    evaluator_server_config = EvaluatorServerConfig(
        custom_port_range=range(49152, 51819)
        if run_model.ert_config.queue_config.queue_system == QueueSystem.LOCAL
        else None
    )

    # Modify the forward model function to track number of calls:
    original_call = run_model.run_forward_model
    run_model.run_forward_model = new_call

    # First run populates the cache:
    run_model.run_experiment(evaluator_server_config)
    assert n_evals > 0
    variables1 = list(run_model.result.controls.values())
    assert np.allclose(variables1, [0.1, 0, 0.4], atol=0.02)

    # Now do another run, where the functions should come from the cache:
    n_evals = 0

    # If we want to do another run, the seba database must be made new:
    Path("everest_output/optimization_output/seba.db").unlink()

    # The batch_id was used as a stopping criterion, so it must be reset:
    run_model.batch_id = 0

    run_model.run_experiment(evaluator_server_config)
    assert n_evals == 0
    variables2 = list(run_model.result.controls.values())
    assert np.array_equal(variables1, variables2)
