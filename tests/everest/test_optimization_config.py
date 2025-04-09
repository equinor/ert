import os

import pytest
from pydantic import ValidationError

from everest.config import EverestConfig


def test_optimization_config(copy_test_data_to_tmp):
    config_directory = "mocked_test_case"
    cfg = os.path.join(config_directory, "config_full_gradient_info.yml")
    full_config_dict = EverestConfig.load_file(cfg)

    optim_config = full_config_dict.optimization
    assert optim_config.algorithm == "optpp_q_newton"
    assert optim_config.perturbation_num == 20
    assert optim_config.max_iterations == 10
    assert optim_config.max_function_evaluations == 1000
    assert optim_config.max_batch_num == 10
    assert optim_config.convergence_tolerance == pytest.approx(1.0e-7)
    assert optim_config.constraint_tolerance == pytest.approx(1.0e-7)
    assert optim_config.speculative is False
    assert optim_config.options == [
        "max_iterations = 0",
        "merit_function el_bakry",
    ]

    assert full_config_dict.model.realizations is not None
    realizations = full_config_dict.model.realizations

    assert realizations == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]


def test_optimization_config_options(copy_test_data_to_tmp):
    config = EverestConfig.load_file("mocked_test_case/mocked_test_case.yml")
    config_dict = config.model_dump(exclude_none=True)

    config_dict["optimization"]["options"] = [
        "max_iterations = 0",
        "merit_function el_bakry",
    ]
    config = EverestConfig.model_validate(config_dict)

    config_dict["optimization"]["options"] = [
        "max_iterations = 0",
        "search_method = 1",
        "merit_function el_bakry",
    ]
    with pytest.raises(
        ValidationError, match=r"Input should be 'value_based_line_search',"
    ):
        config = EverestConfig.model_validate(config_dict)

    config_dict["optimization"]["options"] = [
        "max_iterations = 0",
        "foo = xyz",
        "bar",
        "merit_function el_bakry",
    ]
    with pytest.raises(
        ValidationError, match=r"Unknown or unsupported option\(s\): `foo`, `bar`"
    ):
        config = EverestConfig.model_validate(config_dict)
