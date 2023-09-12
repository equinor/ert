import numpy as np
from hypothesis import given
from hypothesis import strategies as st

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
