import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from scipy.stats import norm

from ert.config import TransferFunction


def valid_params():
    _std = st.floats(min_value=0.01, allow_nan=False, allow_infinity=False)

    mean_min_max_strategy = st.floats(allow_nan=False, allow_infinity=False).flatmap(
        lambda m: st.tuples(
            st.just(m),
            st.floats(m - 2, m - 1),
            st.floats(m + 1, m + 2).filter(
                lambda x: x > m
            ),  # _max, ensuring it's strictly greater than _min
        )
    )

    return mean_min_max_strategy.flatmap(
        lambda triplet: st.tuples(
            st.just(triplet[0]),  # _mean
            _std,  # _std
            st.just(triplet[1]),  # _min
            st.just(triplet[2]),  # _max
        )
    )


@given(st.floats(allow_nan=False, allow_infinity=False), valid_params())
def test_that_truncated_normal_stays_within_bounds(x, arg):
    result = TransferFunction.trans_truncated_normal(x, arg)
    assert arg[2] <= result <= arg[3]


# @reproduce_failure('6.83.0', b'AXicY2BABgfXM2ACRjQalckEAEqXAXY=')
@given(
    st.floats(allow_nan=False, allow_infinity=False),
    st.floats(allow_nan=False, allow_infinity=False),
    valid_params(),
)
def test_that_truncated_normal_is_monotonic(x1, x2, arg):
    arg = (0.0, 2.0, -1.0, 1.0)
    x1 = 0.0
    x2 = 7.450580596923853e-09
    result1 = TransferFunction.trans_truncated_normal(x1, arg)
    result2 = TransferFunction.trans_truncated_normal(x2, arg)
    if np.isclose(x1, x2):
        assert np.isclose(result1, result2, atol=1e-7)
    elif x1 < x2:
        # Results should be different unless clamped
        assert (
            result1 < result2
            or (result1 == arg[2] and result2 == arg[2])
            or (result1 == arg[3] and result2 == arg[3])
        )


@given(valid_params())
def test_that_truncated_normal_is_standardized(arg):
    """If `x` is 0 (i.e., the mean of the standard normal distribution),
    the output should be close to `_mean`.
    """
    result = TransferFunction.trans_truncated_normal(0, arg)
    assert np.isclose(result, arg[0])


@given(st.floats(allow_nan=False, allow_infinity=False), valid_params())
def test_that_truncated_normal_stretches(x, arg):
    """If `x` is 1 standard deviation away from 0, the output should be
    `_mean + _std` or `_mean - _std if `x` is -1.
    """
    if x == 1:
        expected = arg[0] + arg[1]
    elif x == -1:
        expected = arg[0] - arg[1]
    else:
        return
    result = TransferFunction.trans_truncated_normal(x, arg)
    assert np.isclose(result, expected)


def valid_derrf_parameters():
    """All elements in R, min<max, and width>0"""
    steps = st.integers(min_value=2, max_value=1000)
    min_max = (
        st.tuples(
            st.floats(
                min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
            ),
            st.floats(
                min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
            ),
        )
        .map(sorted)
        .filter(lambda x: x[0] < x[1])  # filter out edge case of equality
    )
    skew = st.floats(allow_nan=False, allow_infinity=False)
    width = st.floats(
        min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False
    )
    return min_max.flatmap(
        lambda min_max: st.tuples(
            steps, st.just(min_max[0]), st.just(min_max[1]), skew, width
        )
    )


@given(st.floats(allow_nan=False, allow_infinity=False), valid_derrf_parameters())
def test_that_derrf_is_within_bounds(x, arg):
    """The result shold always be between (or equal) min and max"""
    result = TransferFunction.trans_derrf(x, arg)
    assert arg[1] <= result <= arg[2]


@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=2),
    valid_derrf_parameters(),
)
def test_that_derrf_creates_at_least_steps_or_less_distinct_values(xlist, arg):
    """derrf cannot create more than steps distinct values"""
    res = [TransferFunction.trans_derrf(x, arg) for x in xlist]
    assert len(set(res)) <= arg[0]


@given(st.floats(allow_nan=False, allow_infinity=False), valid_derrf_parameters())
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
    result = TransferFunction.trans_derrf(x, arg)
    assert np.isclose(result, expected)


@given(
    st.tuples(
        st.floats(allow_nan=False, allow_infinity=False),
        st.floats(allow_nan=False, allow_infinity=False),
    )
    .map(sorted)
    .filter(lambda x: x[0] < x[1]),
    valid_derrf_parameters(),
)
def test_that_derrf_is_non_strictly_monotone(x_tuple, arg):
    """`derrf` is a non-strict monotone function"""
    x1, x2 = x_tuple
    assert TransferFunction.trans_derrf(x1, arg) <= TransferFunction.trans_derrf(
        x2, arg
    )
