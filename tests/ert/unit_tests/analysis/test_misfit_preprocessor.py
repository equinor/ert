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
    N = 10000

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

    # To get 1 component, the threshold must be <= the variance
    # of the 1st component.
    # Same for the other components.
    assert get_nr_primary_components(Y, threshold_1 - 0.01) == 1
    assert get_nr_primary_components(Y, threshold_2 - 0.01) == 2
    assert get_nr_primary_components(Y, threshold_3 - 0.01) == 3


@pytest.mark.parametrize("nr_observations", [4, 7, 12])
def test_that_correlated_and_independent_observations_are_grouped_separately(
    nr_observations,
):
    """
    Test the preprocessor's ability to cluster correlated observations
    separately from multiple independent observations.

    We create a response matrix with `nr_observations` rows, where the
    first `nr_observations - 2` rows are strongly correlated, while the
    last two are independent of both the main group and each other.

    This will result in a request for 3 clusters, correctly separating the
    data into its three natural groups:

    1. The main group of correlated observations.
    2. The first independent observation.
    3. The second independent observation.
    """
    rng = np.random.default_rng(1234)
    nr_realizations = 1000
    nr_uncorrelated_obs = 2
    nr_correlated_obs = nr_observations - nr_uncorrelated_obs

    parameters_a = rng.standard_normal(nr_realizations)
    parameters_b = rng.standard_normal(nr_realizations)
    parameters_c = rng.standard_normal(nr_realizations)

    Y = np.zeros((nr_observations, nr_realizations))
    for i in range(nr_correlated_obs):
        Y[i] = (i + 1) * parameters_a
    # The last two observations are independent
    Y[-2] = 10 + parameters_b
    Y[-1] = 5 + parameters_c

    obs_errors = Y.std(axis=1)
    Y_original = Y.copy()
    obs_error_copy = obs_errors.copy()

    scale_factors, clusters, nr_components = main(Y, obs_errors)

    # We expect three distinct clusters now.
    cluster_label_correlated = clusters[0]
    cluster_label_independent_1 = clusters[-2]
    cluster_label_independent_2 = clusters[-1]

    # Check that the three labels are all different
    assert cluster_label_correlated != cluster_label_independent_1
    assert cluster_label_correlated != cluster_label_independent_2
    assert cluster_label_independent_1 != cluster_label_independent_2

    # Check that the main group is clustered together
    for i in range(nr_correlated_obs):
        assert clusters[i] == cluster_label_correlated

    # Correlated group cluster has 1 component.
    # The two independent clusters have size 1, so they also get 1 component.
    # Therefore, all observations should be associated with 1 component.
    expected_nr_components = np.ones(nr_observations, dtype=int)
    np.testing.assert_equal(nr_components, expected_nr_components)

    # For the correlated group: sqrt(num_obs / num_components)
    sf_correlated = np.sqrt(nr_correlated_obs / 1.0)
    # For the independent groups (size 1): sqrt(1 / 1)
    sf_independent = np.sqrt(1.0 / 1.0)

    expected_scale_factors = np.array(
        [sf_correlated] * nr_correlated_obs + [sf_independent] * nr_uncorrelated_obs
    )
    np.testing.assert_allclose(scale_factors, expected_scale_factors)

    np.testing.assert_equal(Y, Y_original)
    np.testing.assert_equal(obs_errors, obs_error_copy)


@pytest.mark.parametrize("nr_observations", [0, 1, 2])
def test_edge_cases_with_few_observations_return_default_values(nr_observations):
    """Test that edge cases with 0-2 observations return default scaling values.
    We do not know why this is the case.
    """
    nr_realizations = 1000
    Y = np.ones((nr_observations, nr_realizations))

    rng = np.random.default_rng(1234)
    parameters_a = rng.normal(10, 1, nr_realizations)

    for i in range(nr_observations):
        Y[i] = (i + 1) * parameters_a

    scale_factors, clusters, nr_components = main(Y, Y.mean(axis=1))

    np.testing.assert_equal(
        scale_factors,
        np.array(nr_observations * [1.0]),
    )

    np.testing.assert_equal(
        clusters,
        np.array(nr_observations * [1.0]),
    )

    np.testing.assert_equal(
        nr_components,
        np.array(nr_observations * [1.0]),
    )


@pytest.mark.parametrize(
    "nr_obs_group_a, nr_obs_group_b",
    [
        (3, 2),
        (5, 5),
        (4, 6),
    ],
)
def test_main_correctly_separates_distinct_correlation_groups(
    nr_obs_group_a, nr_obs_group_b
):
    """
    Creates a response matrix with two distinct and independent groups of
    correlated observations.
    - Group A contains `nr_obs_group_a` responses that are all correlated
      with each other.
    - Group B contains `nr_obs_group_b` responses that are also correlated
      with each other, but are independent of Group A.

    This test asserts that the algorithm places
    the two groups into two separate clusters.
    """
    rng = np.random.default_rng(seed=12345)
    nr_realizations = 1000
    nr_observations = nr_obs_group_a + nr_obs_group_b

    # Create two independent random signals that will form the
    # basis for the two correlation groups.
    params_a = rng.standard_normal(nr_realizations)
    params_b = rng.standard_normal(nr_realizations)

    # Create the final response matrix Y
    Y = np.zeros((nr_observations, nr_realizations))

    # Create Group A: `nr_obs_group_a` perfectly correlated
    # responses based on `params_a`
    for i in range(nr_obs_group_a):
        Y[i] = (i + 1) * params_a

    # Create Group B: `nr_obs_group_b` perfectly correlated
    # responses based on `params_b`
    for i in range(nr_obs_group_b):
        Y[nr_obs_group_a + i] = (i + 1) * params_b

    # Calculate observation errors,
    # required as input for the main function
    obs_errors = Y.std(axis=1)

    scale_factors, clusters, nr_components = main(Y, obs_errors)

    # Assert that the two groups were placed in different clusters.
    # The absolute cluster labels (e.g., 1 vs 2) can change between runs,
    # so we check the grouping structure dynamically.
    cluster_label_group_a = clusters[0]
    cluster_label_group_b = clusters[nr_obs_group_a]

    assert cluster_label_group_a != cluster_label_group_b, (
        "The two distinct correlation groups should be in different clusters."
    )

    # Assert that all members of Group A are in the same cluster
    expected_clusters_a = np.full(nr_obs_group_a, cluster_label_group_a)
    np.testing.assert_array_equal(clusters[:nr_obs_group_a], expected_clusters_a)

    # Assert that all members of Group B are in the same cluster
    expected_clusters_b = np.full(nr_obs_group_b, cluster_label_group_b)
    np.testing.assert_array_equal(clusters[nr_obs_group_a:], expected_clusters_b)

    # Assert the number of components for each observation.
    # Since each group is perfectly correlated internally, the PCA performed
    # on each cluster should find that only 1 principal component is needed.
    expected_nr_components = np.ones(nr_observations, dtype=int)
    np.testing.assert_array_equal(nr_components, expected_nr_components)

    # Assert the calculated scaling factors.
    # The scaling factor is sqrt(num_observations_in_cluster / num_components).
    sf_group_a = np.sqrt(nr_obs_group_a / 1.0)
    sf_group_b = np.sqrt(nr_obs_group_b / 1.0)

    expected_scale_factors = np.array(
        [sf_group_a] * nr_obs_group_a + [sf_group_b] * nr_obs_group_b
    )
    np.testing.assert_allclose(scale_factors, expected_scale_factors)


@pytest.mark.parametrize(
    "nr_obs_group_a, nr_obs_group_b",
    [
        (3, 2),
        (5, 5),
        (4, 6),
    ],
)
@pytest.mark.integration_test
def test_autoscale_clusters_observations_by_correlation_pattern_ignoring_sign(
    nr_obs_group_a, nr_obs_group_b
):
    """
    Creates a response matrix with two distinct and independent groups:

    - Group A contains `nr_obs_group_a` responses that are all positively
      correlated with each other, following the pattern (a+i)*X_1.
    - Group B contains `nr_obs_group_b` responses with alternating signs
      following the pattern (b+j)*(-1)^j*X_2, creating a checkerboard
      correlation pattern within the group, but independent of Group A.

    """
    rng = np.random.default_rng(seed=12345)
    nr_realizations = 1000
    nr_observations = nr_obs_group_a + nr_obs_group_b

    # Create two independent random signals
    X_1 = rng.standard_normal(nr_realizations)
    X_2 = rng.standard_normal(nr_realizations)

    Y = np.zeros((nr_observations, nr_realizations))

    # Create Group A: (a+i)*X_1 pattern - all positively correlated
    a = 1  # base scaling factor for group A
    for i in range(nr_obs_group_a):
        Y[i] = (a + i) * X_1

    # Create Group B: (b+j)*(-1)^j*X_2 pattern - checkerboard correlation
    b = 1  # base scaling factor for group B
    for j in range(nr_obs_group_b):
        sign = (-1) ** j  # alternates: +1, -1, +1, -1, ...
        Y[nr_obs_group_a + j] = (b + j) * sign * X_2

    obs_errors = Y.std(axis=1)
    scale_factors, clusters, nr_components = main(Y, obs_errors)

    # Assert that all members of Group A are in the same cluster
    group_a_clusters = clusters[:nr_obs_group_a]
    assert len(np.unique(group_a_clusters)) == 1, (
        "All Group A responses should be in the same cluster"
    )

    # Assert that all members of Group B are in the same cluster
    group_b_clusters = clusters[nr_obs_group_a:]
    assert len(np.unique(group_b_clusters)) == 1, (
        "All Group B responses should be in the same cluster"
    )

    # Assert that responses from Group A are assigned
    # to a different cluster than responses from Group B
    assert np.unique(group_a_clusters) != np.unique(group_b_clusters)

    # Assert that nr_components are consistent within clusters
    # (all members of the same cluster should have the same nr_components)
    unique_clusters = np.unique(clusters)
    for cluster_id in unique_clusters:
        cluster_indices = np.where(clusters == cluster_id)[0]
        cluster_components = nr_components[cluster_indices]
        assert len(np.unique(cluster_components)) == 1

    # Assert the calculated scaling factors based on actual cluster assignments
    expected_scale_factors = np.zeros(nr_observations)

    for cluster_id in unique_clusters:
        cluster_indices = np.where(clusters == cluster_id)[0]
        cluster_size = len(cluster_indices)
        # Use the actual number of components for this cluster
        cluster_nr_components = nr_components[cluster_indices[0]]
        expected_sf = np.sqrt(cluster_size / float(cluster_nr_components))
        expected_scale_factors[cluster_indices] = expected_sf

    np.testing.assert_allclose(scale_factors, expected_scale_factors)
