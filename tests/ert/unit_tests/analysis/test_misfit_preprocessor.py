import numpy as np
import pytest

from ert.analysis.misfit_preprocessor import (
    get_nr_primary_components,
    get_scaling_factor,
    main,
)


def test_get_scaling_factor():
    assert get_scaling_factor(8, 2) == 2
    assert get_scaling_factor(4, 0) == 2
    assert get_scaling_factor(4, 1) == 2


@pytest.mark.parametrize(
    "p, rho",
    [
        (4, 0.3),
        (10, 0.1),
        (5, 0.8),
    ],
)
@pytest.mark.parametrize("seed", range(9))
def test_that_get_nr_primary_components_is_according_to_theory(p, rho, seed):
    """Based on theory in Multivariate Statistical Methods 4th Edition
    by Donald F. Morrison.
    See section 6.5 - Some Patterned Matrices and Their Principal Components
    on page 283.
    """
    sigma = 1
    N = 100000

    # Define a p x p equicorrelation matrix
    # See Eqn (1)
    Sigma = np.ones(shape=(p, p)) * rho
    np.fill_diagonal(Sigma, sigma**2)

    # The greatest characteristic root of Sigma
    # See Eqn (2)
    # Represents the variance of the first
    # principal component.
    lambda_1 = sigma**2 * (1 + (p - 1) * rho)

    # The remaining p - 1 characteristic roots are all equal to:
    # See Eqn (5)
    lambda_remaining = sigma**2 * (1 - rho)

    # Calculate the theoretical proportion of variance
    # explained directly from the eigenvalues.
    total_variance = lambda_1 + (p - 1) * lambda_remaining
    threshold_1 = lambda_1 / total_variance
    threshold_2 = (lambda_1 + lambda_remaining) / total_variance
    threshold_3 = (lambda_1 + 2 * lambda_remaining) / total_variance

    # Fast sampling of correlated multivariate observations
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(size=(p, N))
    Y = (np.linalg.cholesky(Sigma) @ X).T

    # Adding a bit to the thresholds because of numerical accuracy.
    assert get_nr_primary_components(Y, threshold_1 + 0.01) == 1
    assert get_nr_primary_components(Y, threshold_2 + 0.01) == 2
    assert get_nr_primary_components(Y, threshold_3 + 0.01) == 3


@pytest.mark.parametrize("nr_observations", [4, 7, 12])
@pytest.mark.integration_test
def test_misfit_preprocessor(nr_observations):
    """We create three independent parameters: a, b and c.
    a has multiple observations, which are all strongly correlated
    (of form (1+i)*a, i =(0, 1, ...)), while b and c only has 1 observation.
    We expect the a observations to be scaled in one group,
    and then the b and c observation to be scaled together.
    b and c are scaled together due to the way one chooses pca components:
    max(len([1 for i in variance_ratio[:-1] if i < 0.95]), 1),
    since one does not include the component that crosses the threshold one
    gets one less component then one might expect.
    And the pca components decide the number of clusters for scaling.
    """
    rng = np.random.default_rng(1234)
    nr_realizations = 1000
    nr_correlated_obs = nr_observations - 2
    Y = np.ones((nr_observations, nr_realizations))
    parameters_a = rng.standard_normal(nr_realizations)
    parameters_b = rng.standard_normal(nr_realizations)
    parameters_c = rng.standard_normal(nr_realizations)
    for i in range(nr_correlated_obs):
        Y[i] = (1 + i) * parameters_a
    Y[-1] = 5 + parameters_b
    Y[-2] = 10 + parameters_c
    obs_errors = Y.std(axis=1)
    Y_original = Y.copy()
    obs_error_copy = obs_errors.copy()
    result, *_ = main(Y, obs_errors)
    np.testing.assert_equal(
        result,
        np.array(
            nr_correlated_obs * [np.sqrt(nr_correlated_obs / 1.0)] + 2 * [np.sqrt(2)]
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
    result, *_ = main(Y, Y.mean(axis=1))
    np.testing.assert_equal(
        result,
        np.array(nr_observations * [1.0]),
    )
