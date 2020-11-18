import pytest

from ert3.evaluator import evaluate


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
def test_evaluator(coeffs, expected):
    realizations = [{"coefficients": {"a": a, "b": b, "c": c}} for (a, b, c) in coeffs]
    data = [data["polynomial_output"] for data in evaluate(realizations)]
    assert data == expected


def test_empty_input_evaluator():
    data = evaluate([])
    assert data == []


@pytest.mark.parametrize(
    "realizations, err_msg",
    [
        ([{"coefficients": {"a": 1, "b": 2}}], "only the keys <a>, <b> and <c>"),
        (
            [{"coefficients": {"a": 1, "b": 2, "c": 3, "d": 4}}],
            "only the keys <a>, <b> and <c>",
        ),
        (
            [{"wrong_key": {"a": 1, "b": 2, "c": 3}}],
            "Each entry in the input must be a dict with key",
        ),
        (
            [{"coefficients": "not a dict"}],
            "Each coefficients entry in the input must be a dict, was not a dict",
        ),
        (
            [{"coefficients": {"a": 1, "b": 2, "c": "not a number"}}],
            "each <a>, <b> and <c> must be a number",
        ),
        ("aString", "Each entry in the input must be a dict, was a"),
        ((1), "Input must be an iterable, was 1"),
        (
            ([("single_string")]),
            "Each entry in the input must be a dict, was single_string",
        ),
        ([1, 2, 3], "Each entry in the input must be a dict, was 1"),
    ],
)
def test_incorrect_argument_type(realizations, err_msg):
    with pytest.raises(ValueError, match=err_msg):
        data = evaluate(realizations)
