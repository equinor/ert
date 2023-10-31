import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from ert.analysis import smoother_update
from ert.analysis.misfit_preprocessor import (
    get_nr_primary_components,
    get_scaling_factor,
    main,
)


@pytest.mark.parametrize(
    "nr_obs, nr_components, expected", [[8, 2, 2], [4, 0, 2], [4, 1, 2]]
)
def test_get_scaling_factor(nr_obs, nr_components, expected):
    assert get_scaling_factor(nr_obs, nr_components) == expected


@pytest.mark.parametrize(
    "threshold,expected_result", [(0.0, 0), (0.7, 3), (0.8, 4), (0.9, 5), (0.95, 6)]
)
def test_get_nr_primary_components(threshold, expected_result):
    rng = np.random.default_rng(123)
    input_matrix = rng.random((10, 10))
    components = get_nr_primary_components(input_matrix, threshold)
    assert components == expected_result


def test_that_get_nr_primary_components_is_according_to_theory():
    # pylint: disable=too-many-locals,invalid-name
    """Based on theory in Multivariate Statistical Methods 4th Edition
    by Donald F. Morrison.
    See section 6.5 - Some Patterned Matrices and Their Principal Components.
    """
    rho = 0.3
    p = 4
    sigma = 1
    N = 100000

    R = np.ones(shape=(p, p)) * rho
    np.fill_diagonal(R, sigma**2)

    rng = np.random.default_rng(1234)

    # Fast sampling of correlated multivariate observations
    X = rng.standard_normal(size=(p, N))
    Y = (np.linalg.cholesky(R) @ X).T

    Y = StandardScaler().fit_transform(Y)

    lambda_1 = sigma**2 * (1 + (p - 1) * rho)
    lambda_remaining = sigma**2 * (1 - rho)
    s1 = np.sqrt(lambda_1 * (N - 1))
    s_remaining = np.sqrt(lambda_remaining * (N - 1))

    total = s1**2 + (p - 1) * s_remaining**2
    threshold_1 = s1**2 / total
    threshold_2 = (s1**2 + s_remaining**2) / total
    threshold_3 = (s1**2 + 2 * s_remaining**2) / total

    # Adding a bit to the thresholds because of numerical accuracy.
    components = get_nr_primary_components(Y, threshold_1 + 0.01)
    assert components == 1
    components = get_nr_primary_components(Y, threshold_2 + 0.01)
    assert components == 2
    components = get_nr_primary_components(Y, threshold_3 + 0.01)
    assert components == 3


@pytest.mark.parametrize("nr_observations", [3, 10, 100])
def test_misfit_preprocessor(nr_observations):
    """We create two independent parameters, a and b.
    Using the linear function y = ax.
    a has multiple observations, which are all strongly correlated, while
    b only has 1 observation. We expect the a observations to be scaled,
    and the b observation to be left alone"""
    rng = np.random.default_rng(1234)
    nr_realizations = 1000
    Y = np.ones((nr_observations, nr_realizations))
    parameters_a = rng.standard_normal(nr_realizations)
    parameters_b = rng.standard_normal(nr_realizations)
    for i in range(nr_observations - 1):
        Y[i] = i + 1 * parameters_a
    Y[-1] = 5 + 1 * parameters_b
    obs_errors = Y.mean(axis=1)
    Y_original = Y.copy()
    obs_error_copy = obs_errors.copy()
    result = main(Y, obs_errors)
    np.testing.assert_equal(
        result,
        np.array(
            (nr_observations - 1) * [np.sqrt((nr_observations - 1) / 1.0)] + [1.0]
        ),
    )
    # Check that we don`t modify the input data
    np.testing.assert_equal(Y, Y_original)
    np.testing.assert_equal(obs_errors, obs_error_copy)


@pytest.mark.parametrize("nr_observations", [0, 1, 2])
def test_misfit_preprocessor_single(nr_observations):
    """We create nr_observations responses using the linear function y = ax.
    We don`t have enough information to cluster properly, so we don`t try
    to scale"""
    rng = np.random.default_rng(1234)
    nr_realizations = 1000
    Y = np.ones((nr_observations, nr_realizations))
    parameters_a = rng.normal(10, 1, nr_realizations)
    for i in range(nr_observations):
        Y[i] = (i + 1) * parameters_a
    result = main(Y, Y.mean(axis=1))
    np.testing.assert_equal(
        result,
        np.array(nr_observations * [1.0]),
    )
