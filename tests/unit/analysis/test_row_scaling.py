import random
from functools import partial

import numpy as np
import pytest
from iterative_ensemble_smoother import ESMDA
from iterative_ensemble_smoother.experimental import (
    ensemble_smoother_update_step_row_scaling,
)

from ert.analysis.row_scaling import RowScaling


def row_scaling_coloumb(nx, ny, data_index):
    j = data_index / nx
    i = data_index - j * nx

    dx = 0.5 + i
    dy = 0.5 + j

    r2 = dx * dx + dy * dy
    return 0.50 / r2


def test_length_of_rowscaling_is_initially_zero():
    row_scaling = RowScaling()
    assert len(row_scaling) == 0


def test_row_scaling_automatically_grows_on_set():
    row_scaling = RowScaling()
    row_scaling[9] = 0.25
    assert len(row_scaling) == 10
    assert row_scaling[0] == 0
    assert row_scaling[9] == 0.25


def test_row_scaling_throws_index_error_on_get():
    with pytest.raises(IndexError):
        _ = RowScaling()[10]


def test_assigning_to_index_in_row_scaling_clamps_the_value():
    row_scaling = RowScaling()
    for index, _ in enumerate(row_scaling):
        r = random.random()
        row_scaling[index] = r
        assert row_scaling[index] == row_scaling.clamp(r)


def test_assigning_with_function_applies_function_to_each_index():
    row_scaling = RowScaling()
    row_scaling.assign(100, lambda _: 1)
    assert len(row_scaling) == 100
    for i in range(100):
        row_scaling[i] = 1

    row_scaling.assign(100, lambda x: x / 100)
    for i in range(100):
        assert row_scaling[i] == row_scaling.clamp(i / 100)

    ny = 10
    nx = 10
    coloumb = partial(row_scaling_coloumb, nx, ny)
    row_scaling.assign(nx * ny, coloumb)
    for j in range(ny):
        for i in range(nx):
            g = j * nx + i
            assert row_scaling[g] == row_scaling.clamp(row_scaling_coloumb(nx, ny, g))


def test_assigning_a_non_vector_value_with_assign_vector_raises_value_error():
    row_scaling = RowScaling()
    with pytest.raises(ValueError):
        row_scaling.assign_vector(123.0)


def test_row_scaling_factor_0_for_all_parameters():
    row_scaling = RowScaling()
    A = np.asfortranarray(np.array([[1.0, 2.0], [4.0, 5.0]]))
    observations = np.array([5.0, 6.0])
    S = np.array([[1.0, 2.0], [3.0, 4.0]])

    for i in range(A.shape[0]):
        row_scaling[i] = 0.0

    A_copy = A.copy()
    ((A, row_scaling),) = ensemble_smoother_update_step_row_scaling(
        Y=S,
        X_with_row_scaling=[(A, row_scaling)],
        covariance=np.full(observations.shape, 0.5),
        observations=observations,
    )
    assert np.allclose(A_copy, A)


def test_row_scaling_factor_1_for_either_parameter():
    row_scaling = RowScaling()
    A = np.asfortranarray(np.array([[1.0, 2.0], [4.0, 5.0]]))
    observations = np.array([5.0, 6.0])
    covariance = np.full(observations.shape, 0.5)
    S = np.array([[1.0, 2.0], [3.0, 4.0]])

    # Define row-scaling factors for the two parameters
    row_scaling[0] = 0.0
    row_scaling[1] = 1.0

    # Copy the prior, in case it gets overwritten
    A_prior = A.copy()
    A_no_row_scaling = A.copy()

    # Update with row-scaling
    ((A, row_scaling),) = ensemble_smoother_update_step_row_scaling(
        Y=S,
        X_with_row_scaling=[(A, row_scaling)],
        covariance=covariance,
        observations=observations,
        seed=42,
    )

    # Update without row-scaling, for comparisons
    smoother = ESMDA(
        covariance=covariance,
        observations=observations,
        alpha=np.array([1]),
        seed=42,
    )
    A_no_row_scaling = smoother.assimilate(
        X=A_no_row_scaling,
        Y=S,
    )

    # A[0] should not be updated because row_scaling[0] = 0.0
    # A[1] should get a normal update, because row_scaling[1] = 1.0
    assert np.allclose(A[0], A_prior[0])
    assert np.allclose(A[1], A_no_row_scaling[1])
