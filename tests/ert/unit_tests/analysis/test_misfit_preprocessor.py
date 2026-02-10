import numpy as np
import pytest

from ert.analysis.misfit_preprocessor import (
    cluster_responses,
    get_nr_primary_components,
    get_scaling_factor,
    main,
)


def test_get_scaling_factor():
    assert get_scaling_factor(8, 2) == 2
    assert get_scaling_factor(4, 0) == 2
    assert get_scaling_factor(4, 1) == 2


@pytest.mark.parametrize(
    ("p", "rho"),
    [
        (4, 0.3),
        (10, 0.1),
        (5, 0.8),
    ],
)
@pytest.mark.parametrize("seed", range(4))
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


def test_edge_cases_with_few_observations_return_default_values():
    """
    Test that edge cases with 2 observations return default scaling values.
    We create an example of a response matrix where all rows are perfectly correlated,
    which should lead to a single cluster with 1 component and a scaling factor of
    sqrt(num_observations / num_components) = sqrt(2).
    However, since the number of observations is <= 2, the function should skip the
    clustering and PCA and return default values of 1.0 for all scaling factors
    """
    nr_realizations = 1000
    nr_observations = 2
    Y = np.ones((nr_observations, nr_realizations))

    rng = np.random.default_rng(1234)
    parameters_a = rng.normal(10, 1, nr_realizations)

    # create response matrix
    for i in range(nr_observations):
        Y[i] = (i + 1) * parameters_a

    # Add observation errors
    obs_errors = np.ones(nr_observations) * 0.1

    scale_factors, *_ = main(Y, obs_errors)

    np.testing.assert_equal(
        scale_factors,
        np.array(nr_observations * [1.0]),
    )


@pytest.mark.parametrize(
    ("nr_obs_group_a", "nr_obs_group_b"),
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
    ("nr_obs_group_a", "nr_obs_group_b"),
    [
        (3, 2),
        (5, 5),
        (4, 6),
    ],
)
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


@pytest.mark.parametrize(
    ("corr_de_target", "expect_de_merged"),
    [
        (0.55, True),  # Above threshold ~0.42: D-E merges first
        (0.50, True),
        (0.45, True),
        (0.40, False),  # Below threshold: A-B or B-C merges first
        (0.30, False),
    ],
)
def test_that_clustering_prioritizes_global_similarity_over_local_correlation(
    corr_de_target, expect_de_merged
):
    """
    This test demonstrates that the current clustering implementation (using
    Euclidean distance on correlation rows) can behave unintuitively by prioritizing
    pairs with LOWER direct correlation for merging over pairs with HIGHER direct
    correlation. This happens when the higher-correlation pair disagrees strongly on
    other variables.

    "Prioritizing" here means that the hierarchical clustering algorithm considers
    the lower-correlation pair to be "closer" (more similar) and thus merges them
    earlier in the bottom-up clustering process.

    Scenario:
    - Group 1: A, B, C.
      A and C are independent.
      B = A + C (correlated ~0.707 with both).
      In other words, A and B are correlated ~0.7, the same is true for B and C.
      However, the Euclidean distance between A's and B's correlation rows is large
      because of the disagreement on C

    - Group 2: D, E.
      D and E are isolated and correlated with strength rho (varied by parametrization).
      They agree perfectly on A, B, C (zero correlation with all).
      The Euclidean distance between D's and E's correlation rows is relatively
      small because they have consistent (zero) correlations with everything else.

    Each variable's correlation row (the corresponding row in the correlation matrix)
    includes its correlation with all variables, including itself (1.0 on diagonal).
    For D and E we have:

    D's row: [corr(D,A)=0, corr(D,B)=0, corr(D,C)=0, corr(D,D)=1.0, corr(D,E)=rho]
    E's row: [corr(E,A)=0, corr(E,B)=0, corr(E,C)=0, corr(E,D)=rho, corr(E,E)=1.0]

    For A, B and C we have:
    A's row: [corr(A,A)=1.0, corr(A,B)=0.7, corr(A,C)=0, corr(A,D)=0, corr(A,E)=0]
    B's row: [corr(B,A)=0.7, corr(B,B)=1.0, corr(B,C)=0.7, corr(B,D)=0, corr(B,E)=0]
    C's row: [corr(C,A)=0, corr(C,B)=0.7, corr(C,C)=1.0, corr(C,D)=0, corr(C,E)=0]

    Threshold calculations (based on Euclidean distance between correlation rows):
      dist(D,E) = sqrt((0-0)^2 + (0-0)^2 + (0-0)^2 + (1-rho)^2 + (rho-1)^2))
                = sqrt(2 * (1 - rho)^2)
      dist(A,B) = dist(B,C) = sqrt((1-0.7)^2 + (0.7-1)^2 + (0-0.7)^2 + (0-0)^2+(0-0)^2))
                = 0.82

      Solve dist(D,E) < dist(A,B) for rho:
      sqrt(2 * (1 - rho)^2) < 0.82
        2 * (1 - rho)^2 < 0.82^2
        (1 - rho)^2 < 0.82^2 / 2
        1 - rho < sqrt(0.82^2 / 2)
        rho > 1 - sqrt(0.82^2 / 2) ≈ 0.42
      Hence, when rho > 0.42, D-E merges first; otherwise either A-B or B-C merges first
      (depending on random variation in the sampling).

    This test is parametrized to verify both regimes:
    - rho > 0.42: D-E merges first despite corr(A,B) > rho and corr(B,C) > rho
    - rho < 0.42: D-E no longer the closest pair (either A-B or B-C merges first)
    """

    rng = np.random.default_rng(42)
    N_realizations = 10000

    # Construct A, B, C
    A = rng.standard_normal(N_realizations)
    C = rng.standard_normal(N_realizations)
    # B = A + C. Normalize everyone.
    A = (A - np.mean(A)) / np.std(A)
    C = (C - np.mean(C)) / np.std(C)
    B = A + C
    B = (B - np.mean(B)) / np.std(B)

    # Construct D, E with target correlation
    D = rng.standard_normal(N_realizations)
    noise = rng.standard_normal(N_realizations)
    E = corr_de_target * D + np.sqrt(1 - corr_de_target**2) * noise
    D = (D - np.mean(D)) / np.std(D)
    E = (E - np.mean(E)) / np.std(E)

    dataset = np.array([A, B, C, D, E]).T

    # Verify correlations
    corr = np.corrcoef(dataset, rowvar=False)
    # The return type of corrcoef is ambiguous (float | ndarray) in stubs,
    # so we assert it is an array to silence static analysis warnings about indexing.
    assert isinstance(corr, np.ndarray)
    corr_AB = corr[0, 1]
    corr_BC = corr[1, 2]
    corr_DE = corr[3, 4]

    assert np.isclose(corr_AB, 0.707, atol=0.05), (
        f"Setup error: A-B corr {corr_AB} != 0.707"
    )
    assert np.isclose(corr_BC, 0.707, atol=0.05), (
        f"Setup error: B-C corr {corr_BC} != 0.707"
    )
    assert np.isclose(corr_DE, corr_de_target, atol=0.05), (
        f"Setup error: D-E corr {corr_DE} != {corr_de_target}"
    )

    # We ask for a clustering that would force exactly ONE merge.
    # Total items = 5 (A, B, C, D, E).
    # If we ask for 4 clusters, the algorithm must pick the single "best" pair
    # to merge and leave the others as singletons.
    clusters = cluster_responses(dataset, nr_clusters=4)

    is_DE_merged = clusters[3] == clusters[4]
    is_AB_merged = clusters[0] == clusters[1]
    is_BC_merged = clusters[1] == clusters[2]

    failure_msg = (
        f"DE_merged={is_DE_merged}, AB_merged={is_AB_merged}, BC_merged={is_BC_merged}."
        f"Correlations: AB={corr_AB:.3f}, BC={corr_BC:.3f}, DE={corr_DE:.3f}"
    )
    assert is_DE_merged == expect_de_merged, failure_msg
    # When D-E merges first, A-B and B-Cshould not (they stay separate)
    if expect_de_merged:
        assert not is_AB_merged, failure_msg
        assert not is_BC_merged, failure_msg


def test_clustering_and_scaling_realistic_scenario():
    """
    This is an integration test for two key components of the autoscaler:

    1. Determining the number of clusters.
    2. Determining the scaling factor for each cluster.

    Scenario:
    We are given 500 noisy seismic observations and 20 precise well pressure
    observations. All the seismic observations are correlated and all the
    pressure observations are correlated, but the seismic and the pressure observations
    are independent of each other.

    Intuitive behavior:
    The autoscaler should identify two clusters, one for the seismic observations and
    one for the pressure observations, and assign a scaling factor to each cluster based
    on the number of observations in that cluster (sqrt(500) for seismic and sqrt(20)
    for pressure).

    Actual behavior with current implementation:
    One cluster with scaling factor sqrt(520).

    Explanation of clustering:
    The autoscaler uses PCA to determine the number of clusters. When using PCA it is
    common to normalize the data before calculating the principal components
    (e.g., using StandardScaler to give each observation unit variance). However,
    the current implementation scales the responses by the observation errors instead.
    This means that the precise pressure observations are amplified and the noisy
    seismic observations are suppressed. As a result, the PCA identifies only one
    principal component, and hence, everything is grouped into one cluster.

    Explanation of scaling factor:
    The scaling factor for a cluster is calculated as
    sqrt(num_observations_in_cluster / num_components), where num_components is
    calculated by doing PCA on the elements of the cluster (stopping when 95% of the
    total variance is explained).
    """

    rng = np.random.default_rng(42)
    n_realizations = 100

    n_seismic = 500
    n_pressure = 20

    # Two independent underlying parameters
    param_shallow = rng.normal(0, 1, size=(n_realizations, 1))
    param_deep = rng.normal(0, 1, size=(n_realizations, 1))

    # Initialize data by using broadcasting to create the correlated structures:

    # - Seismic responses = param_shallow * sensitivity,
    #   (all 500 seismic obs are linear combinations of the same parameter to create
    #   within-group correlation)

    # - Pressure responses = param_deep * sensitivity,
    #   (all 20 pressure obs are linear combinations of a different parameter)

    # - Since param_shallow and param_deep are independent, this results in
    #   the seismic responses being independent from the pressure responses.

    # Seismic: sensitive to shallow param
    seismic_sensitivity = rng.uniform(0.5, 1.5, size=(1, n_seismic))
    seismic_responses = param_shallow @ seismic_sensitivity
    # Add noise
    seismic_responses += rng.normal(0, 0.1, size=(n_realizations, n_seismic))

    # Pressure: sensitive to deep param (independent of seismic)
    pressure_sensitivity = rng.uniform(0.5, 1.5, size=(1, n_pressure))
    pressure_responses = param_deep @ pressure_sensitivity
    # Add noise
    pressure_responses += rng.normal(0, 0.1, size=(n_realizations, n_pressure))

    responses = np.hstack([seismic_responses, pressure_responses])

    # Observation errors: seismic is NOISY, pressure is PRECISE
    seismic_errors = np.full(n_seismic, 2.0)
    pressure_errors = np.full(n_pressure, 0.05)
    obs_errors = np.hstack([seismic_errors, pressure_errors])

    # Run the main function to get clusters and scaling factors
    scale_factors, clusters, _ = main(responses.T, obs_errors)

    # Assert that all observations are in the same cluster
    assert len(np.unique(clusters)) == 1

    # Assert that the scaling factor is sqrt(520) for all observations
    # note that this results in the pressure observations receiving a scaling
    # factor of sqrt(520) instead of sqrt(20), which means that they are significantly
    # deflated compared to what one would expect, and essentially ignored/suppressed by
    # the updating algorithm, even though the observations are precise and should be
    # influential.
    expected_sf = np.sqrt(n_seismic + n_pressure)
    assert np.allclose(scale_factors, expected_sf)


def test_clustering_and_scaling_edge_case():
    """
    This test demonstrates that when observations have irregular errors, the
    current clustering approach can lead to unintuitive results where independent
    observations are clustered together, and treated as correlated.

    Scenario:
    Suppose we have 100 independent observations r_1,r_2,...,r_100 with corresponding
    observation errors, where r_1 has a small error, and r_2,...,r_100 have large
    errors. The error-scaling step amplifies r_1's response and suppresses
    r_2,...,r_100, leading to one cluster instead of 100 clusters. As a result, the
    history matching update is deflated as if all 100 observations were
    perfectly correlated, even though they are all independent.
    """

    # Create 100 independent responses
    rng = np.random.default_rng(42)
    n_observations = 100
    n_realizations = 1000
    responses = rng.standard_normal((n_observations, n_realizations))

    # Create irregular observation errors: one small, the rest large
    obs_errors = np.array([0.1] + [10.0] * (n_observations - 1))

    # Run clustering algorithm
    scale_factors, clusters, _ = main(responses, obs_errors)

    # Assert that all observations are clustered together (only 1 cluster)
    assert len(np.unique(clusters)) == 1

    # Assert deflation rate
    # For independent observations, the scaling factor should be 1
    # but since all overvations are clustered together and treated as perfectly
    # correlated, the scaling factor becomes sqrt(100) = 10
    assert len(np.unique(scale_factors)) == 1
    assert np.allclose(scale_factors, np.sqrt(100))
