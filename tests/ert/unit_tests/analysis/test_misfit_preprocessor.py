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


def test_that_get_nr_primary_components_is_according_to_theory():
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

    Y = (Y - Y.mean(axis=0)) / Y.std(axis=0)

    lambda_1 = sigma**2 * (1 + (p - 1) * rho)
    lambda_remaining = sigma**2 * (1 - rho)
    s1 = np.sqrt(lambda_1 * (N - 1))
    s_remaining = np.sqrt(lambda_remaining * (N - 1))

    total = s1**2 + (p - 1) * s_remaining**2
    threshold_1 = s1**2 / total
    threshold_2 = (s1**2 + s_remaining**2) / total
    threshold_3 = (s1**2 + 2 * s_remaining**2) / total

    # Adding a bit to the thresholds because of numerical accuracy.
    assert get_nr_primary_components(Y, threshold_1 + 0.01) == 1
    assert get_nr_primary_components(Y, threshold_2 + 0.01) == 2
    assert get_nr_primary_components(Y, threshold_3 + 0.01) == 3


@pytest.mark.parametrize("nr_observations", [4, 7, 12])
@pytest.mark.integration_test
def test_that_correlated_and_independent_observations_are_grouped_separately(
    nr_observations,
):
    """
    Test the preprocessor's ability to cluster correlated observations.

    We create a response matrix with `nr_observations` rows, where the
    first `nr_observations - 2` rows are strongly correlated, while the
    last two are independent.

    We expect the correlated observations to be scaled in one group,
    and perhaps surprisingly, the two independent observations to be
    scaled together in a second group.
    The reason that the two independent responses end up in the same group
    is due to the way get_nr_primary_components counts PCA components,
    and the fact that the number of PCA components is used as the number
    of clusters.
    It returns the number of components that explain **less** than the
    variance specified as the threshold.
    With a threshold of 0.95, the expression used is as follows:

    max(len([1 for i in variance_ratio[:-1] if i < 0.95]), 1),
    """
    rng = np.random.default_rng(1234)
    nr_realizations = 1000
    nr_uncorrelated_obs = 2
    nr_correlated_obs = nr_observations - nr_uncorrelated_obs

    parameters_a = rng.standard_normal(nr_realizations)
    parameters_b = rng.standard_normal(nr_realizations)
    parameters_c = rng.standard_normal(nr_realizations)

    Y = np.ones((nr_observations, nr_realizations))
    for i in range(nr_correlated_obs):
        Y[i] = (1 + i) * parameters_a
    Y[-1] = 5 + parameters_b
    Y[-2] = 10 + parameters_c

    obs_errors = Y.std(axis=1)
    Y_original = Y.copy()
    obs_error_copy = obs_errors.copy()

    scale_factors, clusters, nr_components = main(Y, obs_errors)

    # Since the first nr_correlated_obs rows of Y are perfectly correlated,
    # we only need one principal component to describe all variance.
    nr_components_correlated_group = 1

    # Because of the way we calculate the number of components
    # (see docstring for details), the two undependent responses
    # are represented by a single component.
    nr_components_uncorrelated_group = 1

    np.testing.assert_equal(
        scale_factors,
        np.array(
            nr_correlated_obs
            * [np.sqrt(nr_correlated_obs / nr_components_correlated_group)]
            + nr_uncorrelated_obs
            * [np.sqrt(nr_uncorrelated_obs / nr_components_uncorrelated_group)]
        ),
    )

    expected_clusters = np.array(nr_correlated_obs * [1] + nr_uncorrelated_obs * [2])
    np.testing.assert_equal(clusters, expected_clusters)

    expected_nr_components = (nr_uncorrelated_obs + nr_correlated_obs) * [1]
    np.testing.assert_equal(nr_components, expected_nr_components)

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
