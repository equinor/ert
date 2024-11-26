# import numpy as np
# from ropt.plan import BasicOptimizer
#
# from ert.storage import open_storage
# from ert.run_models.everest_run_model import EverestRunModel
# from everest.config import EverestConfig, SimulatorConfig
# from everest.optimizer.everest2ropt import everest2ropt
# from everest.simulator import SimulatorCache
# from everest.simulator.everest_to_ert import everest_to_ert_config

CONFIG_FILE = "config_advanced_scipy.yml"


def test_simulator_cache(monkeypatch, copy_math_func_test_data_to_tmp):
    # TODO reimplement test
    pass
    # n_evals = 0
    #
    # def new_call(*args):
    #     nonlocal n_evals
    #     result = original_call(*args)
    #     n_evals += (result.evaluation_ids >= 0).sum()
    #     return result
    #
    # config = EverestConfig.load_file(CONFIG_FILE)
    # config.simulator = SimulatorConfig(enable_cache=True)
    #
    # ropt_config = everest2ropt(config)
    # ert_config = everest_to_ert_config(config)
    # run_model = EverestRunModel.create(config)
    #
    # evaluator_server_config = EvaluatorServerConfig(
    #     custom_port_range=range(49152, 51819)
    #     if run_model.ert_config.queue_config.queue_system == QueueSystem.LOCAL
    #     else None
    # )
    #
    # run_model.run_experiment(evaluator_server_config)
    # original_call = run_model.create_forward_model_evaluator_function()
    #
    # # Run once, populating the cache of the simulator:
    # variables1 = (
    #     BasicOptimizer(
    #         enopt_config=ropt_config,
    #         evaluator=new_call,
    #     )
    #     .run()
    #     .variables
    # )
    # assert variables1 is not None
    # assert np.allclose(variables1, [0.1, 0, 0.4], atol=0.02)
    # assert n_evals > 0
    #
    # # Run again with the same simulator:
    # n_evals = 0
    # variables2 = (
    #     BasicOptimizer(
    #         enopt_config=ropt_config,
    #         evaluator=new_call,
    #     )
    #     .run()
    #     .variables
    # )
    # assert variables2 is not None
    # assert n_evals == 0
    #
    # assert np.array_equal(variables1, variables2)
