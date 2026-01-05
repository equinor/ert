import numpy as np
import pytest

from ert.run_models import MultipleDataAssimilation as mda


@pytest.mark.parametrize(
    ("weights", "expected"),
    [
        ("2, 2, 2, 2", [4] * 4),
        ("1, 2, 4, ", [1.75, 3.5, 7.0]),
        ("1, 0, 1, ", [2, 2]),
        ("1.414213562373095, 1.414213562373095", [2, 2]),
    ],
)
def test_weights(weights, expected):
    weights = mda.parse_weights(weights)
    assert weights == expected
    assert np.reciprocal(weights).sum() == 1.0


def test_invalid_weights():
    with pytest.raises(ValueError):
        mda.parse_weights("2, error, 2, 2")
