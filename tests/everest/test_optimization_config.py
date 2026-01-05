import pytest
from pydantic import ValidationError

from everest.config import OptimizationConfig


def test_optimization_config_with_valid_options():
    OptimizationConfig(
        options=[
            "max_iterations = 0",
            "merit_function el_bakry",
        ]
    )


def test_optimization_config_options_invalid_search_method_value():
    with pytest.raises(
        ValidationError, match=r"Input should be 'value_based_line_search',"
    ):
        OptimizationConfig(
            options=[
                "max_iterations = 0",
                "search_method = 1",
                "merit_function el_bakry",
            ]
        )


def test_optimization_config_options_unknown_options_error():
    with pytest.raises(
        ValidationError, match=r"Unknown or unsupported option\(s\): `foo`, `bar`"
    ):
        OptimizationConfig(
            options=[
                "max_iterations = 0",
                "foo = xyz",
                "bar",
                "merit_function el_bakry",
            ]
        )


@pytest.mark.parametrize(
    ("backend", "algorithm", "expected"),
    [
        (None, None, "optpp_q_newton"),
        ("dakota", None, "dakota/optpp_q_newton"),
        (None, "conmin_mfd", "conmin_mfd"),
        ("dakota", "conmin_mfd", "dakota/conmin_mfd"),
        (None, "slsqp", "slsqp"),
        ("scipy", None, "scipy/default"),
        ("scipy", "slsqp", "scipy/slsqp"),
    ],
)
def test_optimization_config_backend_and_algorithm(backend, algorithm, expected):
    config_dict = {}
    if backend is not None:
        config_dict["backend"] = backend
    if algorithm is not None:
        config_dict["algorithm"] = algorithm
    config = OptimizationConfig.model_validate(config_dict)
    assert config.backend is None
    assert config.algorithm == expected


@pytest.mark.parametrize(
    ("backend", "algorithm", "expected"),
    [
        (None, "foo", "Optimizer algorithm 'foo' not found"),
        (None, "default", "Cannot specify 'default' method without a plugin name"),
        ("foo", None, "Optimizer algorithm 'foo/default' not found"),
        ("foo", "optpp_q_newton", "Optimizer algorithm 'foo/optpp_q_newton' not found"),
    ],
)
def test_optimization_config_backend_and_algorithm_errors(backend, algorithm, expected):
    config_dict = {}
    if backend is not None:
        config_dict["backend"] = backend
    if algorithm is not None:
        config_dict["algorithm"] = algorithm
    with pytest.raises(ValueError, match=expected):
        OptimizationConfig.model_validate(config_dict)
