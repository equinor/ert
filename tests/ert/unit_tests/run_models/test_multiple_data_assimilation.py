import math

import numpy as np
import pytest

from ert.run_models import MultipleDataAssimilation as mda


@pytest.mark.parametrize(
    ("weights", "expected"),
    [
        ("2, 2, 2, 2", [4] * 4),
        ("1, 2, 4, ", [1.75, 3.5, 7.0]),
        ("1.414213562373095, 1.414213562373095", [2, 2]),
    ],
)
def test_that_parse_weights_returns_expected_values(weights, expected):
    weights = mda.parse_weights(weights)
    assert weights == expected
    assert math.isclose(np.reciprocal(weights).sum(), 1.0)


def test_that_non_numeric_weight_raises_value_error():
    with pytest.raises(ValueError, match="could not convert string to float: 'error'"):
        mda.parse_weights("2, error, 2, 2")


@pytest.mark.parametrize(
    "weights",
    [
        "2, -1, 2, 2",
        "2.0, 0.0, 2, 2",
        "-1, -1, -1, 0",
        "0, 0, 0, 0",
        "0.0,1.0, 2.0, 3.0",
    ],
)
def test_that_zero_or_negative_weights_raise_value_error(weights):
    with pytest.raises(
        ValueError,
        match=f"Invalid weights: {weights}. Weights must be positive non zero numbers.",
    ):
        mda.parse_weights(weights)
