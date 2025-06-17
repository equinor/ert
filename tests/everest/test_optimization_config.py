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
