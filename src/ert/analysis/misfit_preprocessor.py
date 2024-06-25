import logging
from typing import Tuple

import numpy as np
import numpy.typing as npt
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


def get_scaling_factor(nr_observations: int, nr_components: int) -> float:
    """
    Calculates an observation scaling factor which is:
        sqrt(nr_obs / pc)
    where:
        nr_obs is the number of observations
        pc is the number of primary components from PCA analysis
            below a user threshold
    """
    logger.info(
        (
            f"Calculation scaling factor, nr of primary components: "
            f"{nr_components}, number of observations: {nr_observations}"
        )
    )
    if nr_components == 0:
        nr_components = 1
        logger.warning(
            (
                "Number of PCA components is 0. "
                "Setting to 1 to avoid division by zero "
                "when calculating scaling factor"
            )
        )

    return np.sqrt(nr_observations / float(nr_components))


def get_nr_primary_components(
    responses: npt.NDArray[np.float64], threshold: float
) -> int:
    """
    Calculate the number of principal components needed to achieve a cumulative
    variance less than a specified threshold using Singular Value Decomposition (SVD).
    """
    data_matrix = responses - responses.mean(axis=0)
    _, singulars, _ = np.linalg.svd(data_matrix.astype(float), full_matrices=False)
    # Calculate cumulative variance ratio:
    # Squared singular values are proportional to variance explained by each principal component.
    # We compute the cumulative sum of these, then divide by their total sum to get the
    # cumulative proportion of variance explained by each successive component.
    variance_ratio = np.cumsum(singulars**2) / np.sum(singulars**2)
    return len([1 for i in variance_ratio[:-1] if i < threshold])


def cluster_responses(
    responses: npt.NDArray[np.float64],
    nr_clusters: int,
) -> npt.NDArray[np.int_]:
    """
    Cluster responses using hierarchical clustering based on Spearman correlation.
    Observations that tend to vary similarly across different simulation runs will be clustered together.
    """
    correlation = spearmanr(responses).statistic
    if isinstance(correlation, np.float64):
        correlation = np.array([[1, correlation], [correlation, 1]])
    linkage_matrix = linkage(correlation, "average", "euclidean")
    return fcluster(linkage_matrix, nr_clusters, criterion="maxclust", depth=2)


def main(
    responses: npt.NDArray[np.float64],
    obs_errors: npt.NDArray[np.float64],
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.int_], npt.NDArray[np.int_]]:
    """
    Perform 'Auto Scaling' to mitigate issues with correlated observations in ensemble smoothers.

    This method was developed internally to address challenges with correlated observations
    in ensemble smoothers, which can lead to overconfident updates. Originally named
    'Misfit Preprocessor', it was renamed to 'Auto Scaling' as it doesn't calculate misfits.

    The procedure involves several steps:

    1. Response Scaling:
        Each response in the data is scaled by its corresponding observation error.
        This is done to balance the responses by their uncertainties.
        By multiplying each response by its corresponding observation error,
        we're weighting the responses based on their uncertainty.
        Responses with larger observation errors (more uncertainty) will
        be scaled up, while those with smaller errors will be scaled down.
        Different types of observations might have different units or scales.
        By scaling them with their respective errors, we're putting all responses
        on a comparable scale, which is important for subsequent clustering and PCA.
    2. PCA for Dimensionality Estimation:
        PCA is performed on the centered and scaled
        responses to estimate the intrinsic dimensionality of the data.
        The number of principal components that cumulatively explain a predefined
        percentage of variance (e.g., 95%) is determined.
    3. Hierarchical Clustering:
        Based on the dimensionality estimated by PCA,
        a hierarchical clustering is performed using the Spearman correlation
        matrix of the scaled responses.
        Clustering is done to identify groups of observations with similar variation patterns.
        Using the number of principal components as the number of clusters is not a widely
        common technique in clustering analysis.
        Potential rationale (the creator of method did not document the reasoning behind the method):
        The number of principal components to explain 95% of the variance can be seen as
        an estimate of the intrinsic dimensionality of the data.
        Using this as the number of clusters assumes that each major direction of variance
        could correspond to a distinct cluster.
        This methods allows the number of clusters to be data-driven rather than pre-determined,
        which can be advantageous in some scenarios.
        One potential issue is that data can have a certain number of significant dimension but
        a different number of natural clusters.
    4. Cluster-Based PCA and Scaling:
        For each cluster, PCA is performed to determine the number of principal components that
        explain a specified percentage of variance within that cluster.
        A scaling factor for the observation errors is then calculated as the square root
        of the ratio of the number of observations in the cluster to the number of principal components.
        The specific formula used is not a standard approach, but may nevertheless be reasonable.
        The ratio of observations to principal components is somewhat analogous to degress of freedom
        in statistical analyses. Scaling by a function of this ratio could be seen as an attempt
        to adjust for the effective sample size.

    Parameters:
    -----------
    responses : npt.NDArray[np.float_]
        2D array of response data. Shape: (n_observations, n_realizations)
    obs_errors : npt.NDArray[np.float_]
        1D array of observation errors. Length: n_observations

    Returns:
    --------
    Tuple[npt.NDArray[np.float_], npt.NDArray[np.int_], npt.NDArray[np.int_]]
        - scale_factors: Array of scaling factors for observation errors
        - clusters: Array of cluster assignments for each observation
        - nr_components: Array of the number of principal components for each observation
    """
    logger.info("Scaling observation errors based on response correlations")
    scale_factors = np.ones(len(obs_errors))
    nr_components = np.ones(len(obs_errors), dtype=int)

    scaled_responses = responses * obs_errors.reshape(-1, 1)

    if len(obs_errors) <= 2:
        # Either observations are not correlated, or only correlated
        # each other
        return scale_factors, np.ones(len(obs_errors), dtype=int), nr_components

    prim_components = get_nr_primary_components(scaled_responses, threshold=0.95)

    clusters = cluster_responses(scaled_responses.T, nr_clusters=prim_components)

    for cluster in np.unique(clusters):
        index = np.where(clusters == cluster)[0]
        if len(index) == 1:
            # Not correlated to anything
            components = 1
        else:
            components = get_nr_primary_components(
                scaled_responses[index], threshold=0.95
            )
            components = 1 if components == 0 else components
        scale_factor = get_scaling_factor(len(index), components)
        nr_components[index] *= components
        scale_factors[index] *= scale_factor
    return scale_factors, clusters, nr_components
