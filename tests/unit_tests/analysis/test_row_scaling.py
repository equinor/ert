import random
from functools import partial

import numpy as np
import pytest
from iterative_ensemble_smoother import ES
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
        S, [(A, row_scaling)], np.full(observations.shape, 0.5), observations
    )
    assert np.all(A == A_copy)


def test_row_scaling_factor_1_for_either_parameter():
    row_scaling = RowScaling()
    A = np.asfortranarray(np.array([[1.0, 2.0], [4.0, 5.0]]))
    observations = np.array([5.0, 6.0])
    S = np.array([[1.0, 2.0], [3.0, 4.0]])
    noise = np.random.rand(*S.shape)

    row_scaling[0] = 0.0
    row_scaling[1] = 1.0

    A_prior = A.copy()
    A_no_row_scaling = A.copy()
    ((A, row_scaling),) = ensemble_smoother_update_step_row_scaling(
        S,
        [(A, row_scaling)],
        np.full(observations.shape, 0.5),
        observations,
        noise=noise,
    )

    smoother = ES()
    smoother.fit(S, np.full(observations.shape, 0.5), observations, noise=noise)
    A_no_row_scaling = smoother.update(A_no_row_scaling)

    assert np.all(A[0] == A_prior[0])
    np.testing.assert_allclose(A[1], A_no_row_scaling[1])
