from warnings import filterwarnings

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from scipy.stats import norm

from ert.config import TransformFunction


@pytest.fixture(autouse=True)
def filter_warnings():
    filterwarnings(
        "ignore",
        message=".*trans_derff suffered from catastrophic loss of precision.*",
        category=UserWarning,
    )
    filterwarnings(
        "ignore", message=".*overflow encountered in.*", category=RuntimeWarning
    )


def nice_floats(*args, **kwargs):
    return st.floats(*args, allow_nan=False, allow_infinity=False, **kwargs)


def valid_truncated_normal_params():
    mean_min_max_strategy = nice_floats().flatmap(
        lambda m: st.tuples(
            st.just(m),
            st.floats(m - 2, m - 1),
            st.floats(m + 1, m + 2).filter(
                lambda x: x > m
            ),  # _max, ensuring it's strictly greater than _mean
        )
    )

    return mean_min_max_strategy.flatmap(
        lambda triplet: st.tuples(
            st.just(triplet[0]),  # _mean
            nice_floats(min_value=0.01),  # _std
            st.just(triplet[1]),  # _min
            st.just(triplet[2]),  # _max
        )
    )


@given(nice_floats(), valid_truncated_normal_params())
def test_that_truncated_normal_stays_within_bounds(x, arg):
    assert arg[2] <= TransformFunction.trans_truncated_normal(x, arg) <= arg[3]


@given(
    st.tuples(
        nice_floats(max_value=1e10),
        nice_floats(max_value=1e10),
    ).map(sorted),
    valid_truncated_normal_params(),
)
def test_that_truncated_normal_is_monotonic(x1x2, arg):
    x1, x2 = x1x2
    assume((x2 - x1) > abs(arg[0] / 1e13) + 1e-13)  # tolerance relative to mean
    result1 = TransformFunction.trans_truncated_normal(x1, arg)
    result2 = TransformFunction.trans_truncated_normal(x2, arg)
    # Results should be different unless clamped
    assert (
        result1 < result2
        or (result1 == arg[2] and result2 == arg[2])
        or (result1 == arg[3] and result2 == arg[3])
    )


@given(valid_truncated_normal_params())
def test_that_truncated_normal_is_standardized(arg):
    """If `x` is 0 (i.e., the mean of the standard normal distribution),
    the output should be close to `_mean`.
    """
    assert np.isclose(TransformFunction.trans_truncated_normal(0, arg), arg[0])


def valid_derrf_parameters():
    """All elements in R, min<max, and width>0"""
    steps = st.integers(min_value=2, max_value=1000)
    min_max = (
        st.tuples(
            nice_floats(min_value=-1e6, max_value=1e6),
            nice_floats(min_value=-1e6, max_value=1e6),
        )
        .map(sorted)
        .filter(lambda x: x[0] < x[1])  # filter out edge case of equality
    )
    skew = nice_floats()
    width = st.floats(
        min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False
    )
    return min_max.flatmap(
        lambda min_max: st.tuples(
            steps, st.just(min_max[0]), st.just(min_max[1]), skew, width
        )
    )


@given(nice_floats(), valid_derrf_parameters())
def test_that_derrf_is_within_bounds(x, arg):
    """The result shold always be between (or equal) min and max"""
    assert arg[1] <= TransformFunction.trans_derrf(x, arg) <= arg[2]


@given(
    st.lists(nice_floats(), min_size=2),
    valid_derrf_parameters(),
)
def test_that_derrf_creates_at_least_steps_or_less_distinct_values(xlist, arg):
    """derrf cannot create more than steps distinct values"""
    assert len({TransformFunction.trans_derrf(x, arg) for x in xlist}) <= arg[0]


@given(nice_floats(), valid_derrf_parameters())
def test_that_derrf_corresponds_scaled_binned_normal_cdf(x, arg):
    """Check correspondance to normal cdf with -mu=_skew and sd=_width"""
    _steps, _min, _max, _skew, _width = arg
    q_values = np.linspace(start=0, stop=1, num=_steps)
    q_checks = np.linspace(start=0, stop=1, num=_steps + 1)[1:]
    p = norm.cdf(x, loc=-_skew, scale=_width)
    bin_index = np.digitize(p, q_checks, right=True)
    expected = q_values[bin_index]
    # scale and ensure ok numerics
    expected = _min + expected * (_max - _min)
    if expected > _max or expected < _min:
        np.clip(expected, _min, _max)
    assert np.isclose(TransformFunction.trans_derrf(x, arg), expected)


@given(
    st.tuples(
        nice_floats(),
        nice_floats(),
    ).map(sorted),
    valid_derrf_parameters(),
)
def test_that_derrf_is_non_strictly_monotone(x_tuple, arg):
    """`derrf` is a non-strict monotone function"""
    x1, x2 = x_tuple
    assert TransformFunction.trans_derrf(x1, arg) <= TransformFunction.trans_derrf(
        x2, arg
    )


def valid_triangular_params():
    # Observed strange behavior for very large floats,
    # so setting min and max values that should cover all possible use-cases.
    mode_min_max_strategy = nice_floats(min_value=-1e12, max_value=1e12).flatmap(
        lambda m: st.tuples(
            st.just(m),  # mode
            st.floats(m - 2, m - 1),  # min
            st.floats(m + 1, m + 2).filter(
                lambda x: x > m
            ),  # max, ensuring it's strictly greater than mode
        )
    )

    return mode_min_max_strategy


@given(nice_floats(), valid_triangular_params())
def test_that_triangular_is_within_bounds(x, args):
    _mode, _min, _max = args
    assert _min <= TransformFunction.trans_triangular(x, [_min, _mode, _max]) <= _max


@given(valid_triangular_params())
def test_mode_behavior(args):
    """
    When the CDF value of x (from the normal distribution) corresponds to the relative position of the mode in the triangular distribution,
    the output of trans_triangular should be the mode (_mode) of the triangular distribution.
    """
    _mode, _min, _max = args
    ymode = (_mode - _min) / (_max - _min)

    x = norm.ppf(ymode)

    assert np.isclose(TransformFunction.trans_triangular(x, [_min, _mode, _max]), _mode)


@given(valid_triangular_params())
def test_that_triangular_is_symmetric_around_mode(args):
    """
    For values of x equidistant from the CDF value at the mode, the outputs should be symmetrically placed around the mode.
    This property holds if the triangular distribution is symmetric.
    """
    _mode, _min, _max = args

    # Ensure the triangular distribution is symmetric
    _mode = (_min + _max) / 2

    ymode = (_mode - _min) / (_max - _min)
    delta = ymode / 2

    # Find x1 and x2 such that their CDF values are equidistant from ymode
    x1 = norm.ppf(ymode - delta)
    x2 = norm.ppf(ymode + delta)

    # Calculate the corresponding triangular values
    y1 = TransformFunction.trans_triangular(x1, [_min, _mode, _max])
    y2 = TransformFunction.trans_triangular(x2, [_min, _mode, _max])

    # Check if y1 and y2 are symmetric around the mode
    assert abs((_mode - y1) - (y2 - _mode)) < 1e-15 * max(
        *map(abs, [x1, x2, _min, _mode, _max])
    )


@given(valid_triangular_params())
def test_that_triangular_is_monotonic(args):
    _mode, _min, _max = args

    ymode = (_mode - _min) / (_max - _min)
    delta = 0.05

    # Test both sides of the mode
    for direction in [-1, 1]:
        # Calculate x values based on the direction (before or after the mode)
        x1 = norm.ppf(ymode + direction * delta)
        x2 = norm.ppf(ymode + direction * 2 * delta)

        y1 = TransformFunction.trans_triangular(x1, [_min, _mode, _max])
        y2 = TransformFunction.trans_triangular(x2, [_min, _mode, _max])

        # Assert monotonicity
        if direction == -1:
            assert y1 >= y2
        else:
            assert y1 <= y2
