"""Waterfall chart data computation for the plotting layer.

Computes per-observation contributions to a scalar parameter update using
the Kalman gain matrix stored as an ensemble blob.
"""

from __future__ import annotations

import io
import logging
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import scipy.sparse

from ert.storage.blob_data import BlobType

if TYPE_CHECKING:
    from ert.storage import Ensemble

logger = logging.getLogger(__name__)


def compute_waterfall_data(
    ensemble: Ensemble,
    parameter_key: str,
    nobservations: int = 10,
) -> pl.DataFrame:
    """Compute waterfall chart data for a scalar parameter.

    This decomposes the update of a single scalar parameter
    (from prior to posterior) into per-observation contributions,
    showing how much each observation "pushed" the parameter during
    the EnIF analysis step.

    Parameters
    ----------
    ensemble : Ensemble
        The posterior ensemble (must have a K matrix blob and a parent).
    parameter_key : str
        The parameter key as shown in the UI (e.g. "GROUP:NAME" or "NAME").
    nobservations : int
        Maximum number of individual observations to show (remainder bundled).

    Returns
    -------
    pl.DataFrame
        Columns: "type" (prior|contribution|posterior), "name", "value".
        Empty DataFrame if data is unavailable.
    """
    # Find K matrix blob representing the Kalman gain
    blobs = ensemble.load_blobs(BlobType.MATRIX)
    k_blob = next((b for b in blobs if b.name == "K"), None)
    if k_blob is None:
        logger.info("No K matrix blob found on ensemble %s", ensemble.id)
        return pl.DataFrame()

    parameter_group_sizes: dict[str, int] = k_blob.blob_info.parameter_group_sizes
    if not parameter_group_sizes:
        logger.info("K blob has no parameter_group_sizes metadata")
        return pl.DataFrame()

    # Resolve parameter config name from the UI key
    config_name = (
        parameter_key.rsplit(":", maxsplit=1)[-1]
        if ":" in parameter_key
        else parameter_key
    )

    if config_name not in parameter_group_sizes:
        logger.info("Parameter %r not found in parameter_group_sizes", config_name)
        return pl.DataFrame()

    # Compute row offset in K for this parameter
    row_offset = 0
    for name, size in parameter_group_sizes.items():
        if name == config_name:
            break
        row_offset += size

    param_size = parameter_group_sizes[config_name]
    if param_size != 1:
        logger.info(
            "Parameter %r has size %d (not scalar), skipping waterfall",
            config_name,
            param_size,
        )
        return pl.DataFrame()

    # Load K matrix
    k_bytes = ensemble.load_blob(k_blob.uri)
    if k_blob.blob_info.sparse:
        k_matrix = scipy.sparse.load_npz(io.BytesIO(k_bytes)).toarray()
    else:
        k_matrix = np.load(io.BytesIO(k_bytes))

    k_row = k_matrix[row_offset, :]

    # Get prior ensemble
    parent_id = ensemble.parent
    if parent_id is None:
        logger.info("Ensemble %s has no parent (prior)", ensemble.id)
        return pl.DataFrame()

    storage = ensemble._storage
    prior = storage.get_ensemble(parent_id)

    # Get active realizations
    ens_mask = prior.get_realization_mask_with_responses()
    iens = np.flatnonzero(ens_mask)
    if len(iens) == 0:
        return pl.DataFrame()

    # Load observations and responses to compute innovation mean
    obs_keys = ensemble.experiment.observation_keys
    if not obs_keys:
        return pl.DataFrame()

    obs_resp_df = prior.get_observations_and_responses(obs_keys, iens)

    # Filter to active observations only
    if "status" in obs_resp_df.columns:
        obs_resp_df = obs_resp_df.filter(obs_resp_df["status"] == "Active")

    observation_values = obs_resp_df["observations"].to_numpy()
    observation_names = obs_resp_df["observation_key"].to_list()

    # Simulated responses: columns named by realization index
    real_cols = [str(i) for i in iens]
    available_cols = [c for c in real_cols if c in obs_resp_df.columns]
    if not available_cols:
        return pl.DataFrame()

    S = obs_resp_df.select(available_cols).to_numpy()
    innovation_mean = observation_values - S.mean(axis=1)

    # Verify dimensions match
    if len(innovation_mean) != k_row.shape[0]:
        logger.warning(
            "Dimension mismatch: K row has %d cols but %d observations",
            k_row.shape[0],
            len(innovation_mean),
        )
        return pl.DataFrame()

    # Load prior and posterior parameter values for the scalar parameter.
    prior_values = prior.load_parameters_numpy(config_name, iens)
    posterior_values = ensemble.load_parameters_numpy(config_name, iens)

    prior_flat = prior_values.flatten()
    posterior_flat = posterior_values.flatten()
    prior_mean_raw = float(prior_flat.mean())
    prior_std = float(prior_flat.std())

    if prior_std < 1e-12:
        # If the prior has no variability, we can only report the raw shift.
        param_prior = 0.0
        param_posterior = float(posterior_flat.mean() - prior_mean_raw)
    else:
        param_prior = 0.0
        param_posterior = float((posterior_flat.mean() - prior_mean_raw) / prior_std)

    # Compute scaled contributions that sum to the standardized mean update.
    k_scaled = k_row * innovation_mean
    total_raw = float(k_row @ innovation_mean)
    factor = param_posterior / total_raw if abs(total_raw) > 1e-15 else 0.0

    contributions = factor * k_scaled

    # Sort by absolute contribution, take top N
    sorted_indices = np.argsort(np.abs(contributions))[::-1]
    sorted_names = np.array(observation_names)[sorted_indices]
    sorted_values = contributions[sorted_indices]

    if len(sorted_values) > nobservations:
        top_values = sorted_values[:nobservations]
        top_names = list(sorted_names[:nobservations])
        bundled_value = float(sorted_values[nobservations:].sum())
        top_values = np.append(top_values, bundled_value)
        top_names.append("Other observations")
    else:
        top_values = sorted_values
        top_names = list(sorted_names)

    # Build result DataFrame
    types = ["prior", *(["contribution"] * len(top_values)), "posterior"]
    names = ["Prior", *top_names, "Posterior"]
    values = [param_prior, *top_values.tolist(), param_posterior]

    return pl.DataFrame({"type": types, "name": names, "value": values})
