# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Test new variance inflation algorithm based on KMeans on Drogon data (or other data from ert's storage)
#
# 1. Run ert on Drogon or on some other case
# 2. Update variable `storage_path` to point to ert's internal storage
# 3. Notebook can now be run to both run the currently implemented autoscaling algorithm and the new one based on KMeans

# %%
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.stats import spearmanr
from sklearn.cluster import KMeans

from ert.storage import open_storage

# %% [markdown]
# # List all available experiments in storage

# %%
# storage_path = "01_drogon_ahm/storage/"
storage_path = "/Users/FCUR/git/ert/test-data/ert/poly_example/storage"
with open_storage(storage_path) as storage:
    [print(f"Experiment names: {x.name}") for x in storage.experiments]

# %% [markdown]
# # Pick which experiment to analyse

# %%
experiment_name = "ensemble_smoother"

# %% [markdown]
# # Load observations and responses from storage. Remove responses with zero standard deviation.

# %%
with open_storage(storage_path, "r") as storage:
    ensemble = storage.get_experiment_by_name(experiment_name).get_ensemble_by_name(
        "default_0"
    )
    selected_obs = ensemble.experiment.observation_keys
    iens_active_index = np.flatnonzero(ensemble.get_realization_list_with_responses())
    observations_and_responses = ensemble.get_observations_and_responses(
        selected_obs, iens_active_index
    )

response_cols = [str(i) for i in range(1, ensemble.ensemble_size)]
df_filtered = observations_and_responses.filter(
    pl.concat_list([pl.col(col) for col in response_cols])
    .list.eval(pl.element().std())
    .list.first()
    > 0
)

# %% [markdown]
# # Add column with normalized misfit and sort by it

# %%
# First calculate the misfit column
df_filtered = df_filtered.with_columns(
    misfit_normalized=(
        pl.concat_list(
            [
                (pl.col(col) - pl.col("observations")).pow(2) / pl.col("std").pow(2)
                for col in response_cols
            ]
        )
        .list.sum()  # Sum the squares
        .sqrt()  # Take square root
    )
)

# Get the current column order and insert misfit_normalized after std
columns = df_filtered.columns
std_index = columns.index("std")
new_order = (
    columns[: std_index + 1]  # Everything up to and including 'std'
    + ["misfit_normalized"]  # Insert misfit_normalized
    + [
        col for col in columns[std_index + 1 :] if col != "misfit_normalized"
    ]  # Rest of columns
)

# Reorder and sort
df_filtered = df_filtered.select(new_order).sort("misfit_normalized", descending=True)

df_filtered

# %% [markdown]
# # Plot responses with high misfit
#
# **Question:** What is a good measure of `coverage`?

# %%
# How many to plot
n_misfits = 5
top_misfits = df_filtered.top_k(n_misfits, by="misfit_normalized")

# Create subplots for the top 3 misfits
fig, axes = plt.subplots(n_misfits, 1, figsize=(12, 15))
fig.suptitle(f"Top {n_misfits} Largest Misfits", fontsize=16, fontweight="bold")

for i, ax in enumerate(axes):
    # Get data for current row
    response_key = top_misfits["response_key"][i]
    responses = top_misfits.select(top_misfits.columns[5:-1]).row(
        i
    )  # Exclude the new misfit_normalized column
    observation = top_misfits["observations"][i]
    error = top_misfits["std"][i]
    misfit_norm = top_misfits["misfit_normalized"][i]

    # Plot ensemble responses with better styling
    ax.plot(
        responses,
        "o-",
        alpha=0.7,
        linewidth=1,
        markersize=4,
        color="steelblue",
        label="Ensemble responses",
    )

    # Add observation and uncertainty bounds with clear styling
    ax.axhline(
        observation,
        color="red",
        linewidth=2,
        linestyle="-",
        label=f"Observation: {observation:.3f}",
    )
    ax.axhline(
        observation + error,
        color="red",
        linewidth=1,
        linestyle="--",
        alpha=0.7,
        label="±1σ bounds",
    )
    ax.axhline(observation - error, color="red", linewidth=1, linestyle="--", alpha=0.7)

    # Fill uncertainty region
    ax.fill_between(
        range(len(responses)),
        observation - error,
        observation + error,
        alpha=0.2,
        color="red",
        label="Uncertainty band",
    )

    # Improve labels and title
    ax.set_title(f"#{i + 1} Misfit: {response_key}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Realization Index", fontsize=10)
    ax.set_ylabel("Response Value", fontsize=10)

    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    # Add some statistics as text
    mean_response = np.mean(responses)
    ax.text(
        0.02,
        0.98,
        f"Mean ensemble: {mean_response:.3f}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        fontsize=9,
    )

plt.tight_layout()

# %% [markdown]
# # Run auto scale exactly as it is implemented in ert
#
# TODO: I think I need to include the recent bug-fixes here.

# %%
Y = df_filtered.select(pl.col(response_cols)).to_numpy()

# Create figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# First subplot - Unscaled responses
ax1.set_title("Unscaled responses")
ax1.set_xlabel("Realization")
ax1.set_ylabel("Response")
for i in range(Y.shape[0]):
    ax1.plot(Y[i, :], alpha=0.3)

# Calculate scaled responses
Y_std = (Y - Y.mean(axis=1, keepdims=True)) / Y.std(axis=1, keepdims=True)

# Second subplot - Scaled responses
ax2.set_title("Scaled responses")  # Fixed typo: "set_tiel" -> "set_title"
ax2.set_xlabel("Realization")
ax2.set_ylabel("Response")
for i in range(Y_std.shape[0]):
    ax2.plot(Y_std[i, :], alpha=0.3)

# Adjust layout to prevent overlap
plt.tight_layout()

_, s_std, _ = np.linalg.svd(Y_std.T, full_matrices=False)

var_ratio_std = np.cumsum(s_std**2) / np.sum(s_std**2)

threshold = 0.95
# TODO: There's a PR that improves the way we count pca components
nr_components_std = max(len([1 for i in var_ratio_std[:-1] if i < threshold]), 1)

fig, ax = plt.subplots()
ax.set_title("Variance explained")
ax.plot(range(1, len(var_ratio_std) + 1), var_ratio_std, linestyle="-", linewidth=2)
ax.axvline(nr_components_std, linestyle="--", alpha=0.7, label="95% variance")
ax.annotate(
    f"{nr_components_std} components\nexplain 95% var", xy=(nr_components_std + 1, 0.7)
)
ax.legend()

correlation = spearmanr(Y_std.T).statistic

linkage_matrix = linkage(correlation, "average", "euclidean")

cluster_labels = fcluster(linkage_matrix, nr_components_std, criterion="maxclust")

fig, ax = plt.subplots()
ax.hist(
    cluster_labels, bins=range(1, nr_components_std + 2), alpha=0.7, edgecolor="black"
)
ax.set_xlabel("Cluster Label")
ax.set_ylabel("Number of Observations")
ax.set_title(f"Distribution of Cluster Assignments (k={nr_components_std})")
ax.set_xticks(range(1, nr_components_std + 1))
ax.grid(True, alpha=0.3)


# %% [markdown]
# # Alternative method using Kmeans and misfits


# %%
def find_optimal_clusters_elbow(data):
    """
    Find optimal number of clusters using the "elbow method"

    The elbow method works like this:
    1. Try different numbers of clusters (2, 3, 4, etc.)
    2. For each number, measure how "tight" the clusters are (inertia)
    3. Plot inertia vs number of clusters - it usually looks like an arm
    4. The "elbow" (bend) in this curve is the optimal number

    Think of it like: "How many groups do I need before adding more groups
    doesn't help much?"
    """
    max_k = min(10, len(data) // 5)  # Don't try more clusters than we have data points
    K_range = range(2, max_k + 1)  # Try 2, 3, 4, ... clusters

    inertias = []  # This will store how "spread out" each clustering solution is
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data.reshape(-1, 1))
        inertias.append(
            kmeans.inertia_
        )  # inertia = sum of distances from points to cluster centers

    # Find the "elbow" - where the improvement starts slowing down
    if len(inertias) >= 3:
        diffs = np.diff(inertias)  # How much inertia decreases each step
        diff2 = np.diff(diffs)  # How much the decrease rate changes
        elbow_idx = np.argmax(diff2)  # Find biggest change in decrease rate
        optimal_k = K_range[elbow_idx + 1]
    else:
        optimal_k = 3  # If we don't have enough data points, just use 3 clusters

    return optimal_k


def cluster_by_misfit_kmeans_auto(observed, predicted, measurement_std):
    """
    Automatically group responses based on how badly the model fits them.

    CORE ASSUMPTION:
    ================
    Responses with similar misfit patterns are affected by the same underlying
    model deficiencies (e.g., wrong permeability zones, missing physics, incorrect
    boundary conditions) and should therefore be grouped together for similar
    inflation treatment.

    The Method:
    ===========
    Some data points fit our model well (small residuals), others fit poorly
    (large residuals). We group responses with similar "badness of fit" together,
    then apply different scaling factors to each group:
    - Low misfit groups: Model works well → trust data highly (low inflation)
    - High misfit groups: Systematic model error → reduce trust (high inflation)

    Parameters:
    -----------
    observed : array - what we actually measured in the field
    predicted : array - what our model predicted we should measure
    measurement_std : array - how accurate our measurements are (measurement error)

    Returns:
    --------
    cluster_labels : array - which group each data point belongs to
    inflation_factors : dict - how much to scale down the importance of each group
    n_clusters : int - how many groups we found

    Usage in History Matching:
    ==========================
    Use the inflation_factors to scale measurement error standard deviations:
    inflated_std[i] = measurement_std[i] * inflation_factors[cluster_labels[i]]

    This reduces the weight of systematically misfitting data during parameter
    updates, preventing spurious uncertainty reduction from imperfect models.
    """

    # Step 1: Calculate how bad each prediction is, accounting for measurement accuracy
    #
    # Think of this like: "How many standard deviations off is my prediction?"
    # If observed=100, predicted=90, measurement_std=5:
    # standardized_residual = (100-90)/5 = 2 standard deviations off
    # We square it to make all values positive and emphasize big errors
    raw_residuals = (observed - predicted) / measurement_std  # How many std devs off
    standardized_residuals = raw_residuals**2  # Square to get chi-squared

    # Step 2: Automatically figure out how many groups we need
    # Some responses will fit well (small residuals), others poorly (big residuals)
    # The elbow method finds natural groupings in this "badness of fit"
    n_clusters = find_optimal_clusters_elbow(standardized_residuals)

    # Step 3: Group responses by similar "badness of fit"
    # K-means will find groups where responses within each group have similar
    # levels of model error - this implements our core assumption that similar
    # misfits indicate similar underlying model deficiencies
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(standardized_residuals.reshape(-1, 1))

    # Step 4: Calculate scaling factors for each group based on cluster size
    #
    # The idea: Responses clustered by similar misfit are affected by the same
    # systematic model error and therefore contain redundant information.
    # We inflate their variance to prevent this redundant information from
    # over-constraining the model parameters.
    inflation_factors = {}
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_size = np.sum(cluster_mask)

        # Use square root of cluster size as inflation factor
        # This treats clustered responses as having reduced effective degrees of freedom
        # - Small clusters (size 2-3): minimal inflation (α ≈ 1.4-1.7)
        # - Large clusters (size 16+): significant inflation (α ≥ 4.0)
        inflation_factors[cluster_id] = max(1.1, np.sqrt(cluster_size))

    return cluster_labels, inflation_factors, n_clusters


# Usage example:
observations = df_filtered["observations"].to_numpy()
std = df_filtered["std"].to_numpy()
predicted_values = Y.mean(axis=1)

# Run the automatic clustering
cluster_labels, inflation_factors, n_clusters = cluster_by_misfit_kmeans_auto(
    observed=observations, predicted=predicted_values, measurement_std=std
)

print(f"Elbow method selected {n_clusters} clusters")
print("\nInflation factors by cluster:")
print("(Higher factors = model fits worse = trust this data less)")
print()

for cluster_id, factor in inflation_factors.items():
    n_responses = np.sum(cluster_labels == cluster_id)
    cluster_mask = cluster_labels == cluster_id

    # Recalculate chi-squared for display
    standardized_residuals = ((observations - predicted_values) / std) ** 2
    mean_chi2 = np.mean(standardized_residuals[cluster_mask])

    print(f"Cluster {cluster_id}: α={factor:.2f} ({n_responses} responses)")
    print()

# %%
