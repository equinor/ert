import pytest

from ert3.evaluator import evaluate
from ert3.evaluator.poly import polynomial


@pytest.mark.parametrize(
    "coeffs, expected",
    [
        ([(0, 0, 0)], [[0] * 10]),
        (
            [(1.5, 2.5, 3.5)],
            [[3.5, 7.5, 14.5, 24.5, 37.5, 53.5, 72.5, 94.5, 119.5, 147.5]],
        ),
        (
            [(1.5, 2.5, 3.5), (5, 4, 3)],
            [
                [3.5, 7.5, 14.5, 24.5, 37.5, 53.5, 72.5, 94.5, 119.5, 147.5],
                [3, 12, 31, 60, 99, 148, 207, 276, 355, 444],
            ],
        ),
    ],
)
def test_evaluator(coeffs, expected, tmpdir):
    with tmpdir.as_cwd():
        realizations = [
            {"coefficients": {"a": a, "b": b, "c": c}} for (a, b, c) in coeffs
        ]
        data = [
            data["polynomial_output"] for data in evaluate(realizations, polynomial)
        ]
        assert data == expected
