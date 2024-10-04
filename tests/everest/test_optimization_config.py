import os

import pytest

from everest.config import EverestConfig


def test_optimization_config(copy_test_data_to_tmp):
    config_directory = "mocked_test_case"
    cfg = os.path.join(config_directory, "config_full_gradient_info.yml")
    full_config_dict = EverestConfig.load_file(cfg)

    optim_config = full_config_dict.optimization
    assert optim_config.algorithm == "conmin_mfd"
    assert optim_config.perturbation_num == 20
    assert optim_config.max_iterations == 10
    assert optim_config.max_function_evaluations == 1000
    assert optim_config.max_batch_num == 10
    assert optim_config.convergence_tolerance == pytest.approx(1.0e-7)
    assert optim_config.constraint_tolerance == pytest.approx(1.0e-7)
    assert optim_config.speculative is False
    assert optim_config.options == [
        "131.3 = No",
        "LOG HIGH",
        "Dostuff: 1",
        "speculate",
    ]

    assert full_config_dict.model.realizations is not None
    realizations = full_config_dict.model.realizations

    assert realizations == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
