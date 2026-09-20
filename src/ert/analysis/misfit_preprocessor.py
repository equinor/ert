import logging

import numpy as np
import numpy.typing as npt
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


def get_scaling_factor(nr_observations: int, nr_components: int) -> float:
    """Calculate the observation-error scaling factor.

    Args:
        nr_observations is the number of observations
        nr_components is the number of principal components retained from PCA
    """
    if nr_components == 0:
        nr_components = 1
        logger.warning(
            "Number of PCA components is 0. "
            "Setting to 1 to avoid division by zero "
            "when calculating scaling factor"
        )

    return np.sqrt(nr_observations / float(nr_components))


def get_nr_primary_components(
    responses: npt.NDArray[np.float64], threshold: float
) -> int:
    """
    Calculate the number of principal components required
    to explain a given amount of variance in the responses.

    Args:
    responses: A 2D array of data with shape
        (n_realizations, n_observations).
    threshold: The cumulative variance threshold to meet or exceed.
        For example, a value of 0.95 will find the number of
        components needed to explain at least 95% of the total variance.

    Returns:
        The minimum number of principal components required to meet or exceed
        the specified variance threshold.
    """
    data_matrix = responses - responses.mean(axis=0)
    _, singulars, _ = np.linalg.svd(data_matrix.astype(float), full_matrices=False)
    # Calculate cumulative variance ratio:
    # Squared singular values are proportional to variance explained by each principal
    # component. We compute the cumulative sum of these, then divide by their total
    # sum to get the cumulative proportion of variance explained by each successive
    # component.
    variance_ratio = np.cumsum(singulars**2) / np.sum(singulars**2)

    num_components = np.searchsorted(variance_ratio, threshold, side="left") + 1

    return int(num_components)


def cluster_responses(
    responses: npt.NDArray[np.float64],
    nr_clusters: int,
) -> npt.NDArray[np.int_]:
    """
    Cluster responses using hierarchical clustering on absolute Spearman correlation.
    Observations that vary similarly across realizations, regardless of whether
    the relationship is positive or negative, will tend to be clustered together.
    """
    correlation = spearmanr(responses).statistic
    if isinstance(correlation, np.float64):
        correlation = np.array([[1, correlation], [correlation, 1]])
    # Take absolute value to cluster based on correlation strength rather
    # than direction.
    # This ensures that strong negative correlations (-0.9) are
    # treated as similar to
    # strong positive correlations (+0.9), since both represent
    # strong relationships.
    correlation = np.abs(correlation)
    linkage_matrix = linkage(correlation, "average", "euclidean")
    return fcluster(linkage_matrix, nr_clusters, criterion="maxclust")


def main(
    responses: npt.NDArray[np.float64],
    obs_errors: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int_], npt.NDArray[np.int_]]:
    """
    Perform 'Auto Scaling' to mitigate issues with correlated observations in ensemble
    smoothers.

    This method was developed internally to address challenges with correlated
    observations in ensemble smoothers, which can lead to overconfident updates.
    Originally named 'Misfit Preprocessor', it was renamed to 'Auto Scaling' as
    it doesn't calculate misfits.

    The procedure involves several steps:

    1. Response Scaling:
        Each response is centered by subtracting its ensemble mean and then
        divided by its ensemble standard deviation. This removes offsets and
        puts all responses on a common unit-variance scale before the PCA and
        clustering steps.
    2. PCA for Dimensionality Estimation:
        PCA is performed on the centered and scaled
        responses to estimate the intrinsic dimensionality of the data.
        The number of principal components that cumulatively explain a predefined
        percentage of variance (e.g., 95%) is determined.
    3. Hierarchical Clustering:
        Based on the dimensionality estimated by PCA,
        a hierarchical clustering is performed using the Spearman correlation
        matrix of the scaled responses.
        Clustering is done to identify groups of observations with similar variation
        patterns. Using the number of principal components as the number of clusters
        is not a widely common technique in clustering analysis.
        Potential rationale (the creator of method did not document the reasoning
        behind the method):
        The number of principal components to explain 95% of the variance can be seen as
        an estimate of the intrinsic dimensionality of the data.
        Using this as the number of clusters assumes that each major direction of
        variance could correspond to a distinct cluster.
        This method allows the number of clusters to be data-driven rather than
        pre-determined, which can be advantageous in some scenarios.
        One potential issue is that data can have a certain number of significant
        dimensions but a different number of natural clusters.
    4. Cluster-Based PCA and Scaling:
        For each cluster, PCA is performed to determine the number of principal
        components that explain a specified percentage of variance within that cluster.
        A scaling factor for the observation errors is then calculated as the square
        root of the ratio of the number of observations in the cluster to the number
        of principal components. The specific formula used is not a standard approach,
        but may nevertheless be reasonable. If N responses are perfectly correlated,
        the number of principal components will be 1, resulting in a scaling factor
        of sqrt(N). According to statistical theory, this is the correct scaling factor
        to use when N responses are perfectly correlated, as it ensures the accumulated
        weight (precision) of the N copies equals that of a single independent
        observation. The term sqrt(N/N_eff), where N_eff is the number of principal
        components (effective independent samples), can be interpreted as an
        adjustment for the effective sample size.

    Parameters:
    -----------
    responses : npt.NDArray[np.float_]
        2D array of response data. Shape: (n_observations, n_realizations)
    obs_errors : npt.NDArray[np.float_]
        1D array of observation errors. Only the length is used here to size the
        outputs and to short-circuit when there are two or fewer observations;
        the returned scale factors are applied to the observation errors by the
        caller.

    Returns:
    --------
    Tuple[npt.NDArray[np.float_], npt.NDArray[np.int_], npt.NDArray[np.int_]]
        - scale_factors: Array of scaling factors for observation errors
        - clusters: Array of cluster assignments for each observation
        - nr_components: Array of the number of principal components for each
          observation
    """
    scale_factors = np.ones(len(obs_errors))
    nr_components = np.ones(len(obs_errors), dtype=int)
    scaled_responses = (
        responses - responses.mean(axis=1).reshape(-1, 1)
    ) / responses.std(axis=1).reshape(-1, 1)

    if len(obs_errors) <= 2:
        logger.info("Skipping auto scaling for two or fewer observations")
        return scale_factors, np.ones(len(obs_errors), dtype=int), nr_components

    prim_components = get_nr_primary_components(scaled_responses.T, threshold=0.95)
    clusters = cluster_responses(scaled_responses.T, nr_clusters=prim_components)

    for cluster in np.unique(clusters):
        index = np.where(clusters == cluster)[0]
        if len(index) == 1:
            # A singleton cluster contributes one effective component.
            components = 1
        else:
            components = get_nr_primary_components(
                scaled_responses[index].T, threshold=0.95
            )
            components = 1 if components == 0 else components
        scale_factor = get_scaling_factor(len(index), components)
        nr_components[index] *= components
        scale_factors[index] *= scale_factor
    logger.info(
        f"Calculated scaling factors for {len(scale_factors)} observations across "
        f"{len(np.unique(clusters))} clusters"
    )
    return scale_factors, clusters, nr_components
