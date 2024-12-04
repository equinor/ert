import numpy as np
from ropt.plan import BasicOptimizer

from ert.storage import open_storage
from everest.config import EverestConfig, SimulatorConfig
from everest.optimizer.everest2ropt import everest2ropt
from everest.simulator import Simulator
from everest.simulator.everest_to_ert import everest_to_ert_config

CONFIG_FILE = "config_advanced_scipy.yml"


def test_simulator_cache(monkeypatch, copy_math_func_test_data_to_tmp):
    n_evals = 0
    original_call = Simulator._run_forward_model

    def new_call(*args):
        nonlocal n_evals
        result = original_call(*args)
        n_evals += (result.evaluation_ids >= 0).sum()
        return result

    monkeypatch.setattr(Simulator, "_run_forward_model", new_call)

    config = EverestConfig.load_file(CONFIG_FILE)
    config.simulator = SimulatorConfig(enable_cache=True)

    ropt_config = everest2ropt(config)
    ert_config = everest_to_ert_config(config)

    with open_storage(ert_config.ens_path, mode="w") as storage:
        simulator = Simulator(config, ert_config, storage)

        # Run once, populating the cache of the simulator:
        variables1 = (
            BasicOptimizer(
                enopt_config=ropt_config,
                evaluator=simulator.create_forward_model_evaluator_function(),
            )
            .run()
            .variables
        )
        assert variables1 is not None
        assert np.allclose(variables1, [0.1, 0, 0.4], atol=0.02)
        assert n_evals > 0

        # Run again with the same simulator:
        n_evals = 0
        variables2 = (
            BasicOptimizer(
                enopt_config=ropt_config,
                evaluator=simulator.create_forward_model_evaluator_function(),
            )
            .run()
            .variables
        )
        assert variables2 is not None
        assert n_evals == 0

        assert np.array_equal(variables1, variables2)
