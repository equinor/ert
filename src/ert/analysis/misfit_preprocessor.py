import logging

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
    responses: npt.NDArray[np.float_], threshold: float
) -> int:
    """
    Takes a matrix, does PCA and calculates the cumulative variance ratio
    and returns an int which is the number of primary components where
    the cumulative variance is smaller than user set threshold.
    Also returns an array of singular values.
    """
    data_matrix = responses - responses.mean(axis=0)
    _, singulars, _ = np.linalg.svd(data_matrix.astype(float), full_matrices=False)
    variance_ratio = np.cumsum(singulars**2) / np.sum(singulars**2)
    return len([1 for i in variance_ratio[:-1] if i < threshold])


def cluster_responses(
    responses: npt.NDArray[np.float_],
    nr_components: int,
) -> npt.NDArray[np.int_]:
    """
    Given responses and nr of clusters, returns clusters.
    """
    correlation = spearmanr(responses).statistic
    if isinstance(correlation, np.float64):
        correlation = np.array([[1, correlation], [correlation, 1]])
    linkage_matrix = linkage(correlation, "single", "euclidean")
    clusters = fcluster(
        linkage_matrix, nr_components, criterion="inconsistent", depth=2
    )
    return clusters


def main(
    responses: npt.NDArray[np.float_], obs_errors: npt.NDArray[np.float_]
) -> npt.NDArray[np.float_]:
    logger.info("Scaling observation errors based on response correlations")
    scale_factors = np.ones(len(obs_errors))

    normalized_responses = (responses.T * obs_errors).T
    if len(obs_errors) <= 2:
        # Either observations are not correlated, or only correlated
        # each other
        return scale_factors
    nr_components = get_nr_primary_components(normalized_responses, threshold=0.95)
    clusters = cluster_responses(normalized_responses.T, nr_components)

    for cluster in np.unique(clusters):
        index = np.where(clusters == cluster)[0]
        if len(index) == 1:
            # Not correlated to anything, so we can continue
            continue
        nr_components = get_nr_primary_components(
            normalized_responses[index], threshold=0.95
        )
        scale_factor = get_scaling_factor(len(index), nr_components)
        scale_factors[index] *= scale_factor
    return scale_factors
