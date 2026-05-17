from typing import Any
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest
import xarray as xr
from graphite_maps.linear_regression import (  # type: ignore
    linear_boost_ic_regression,
)

from ert.analysis._enif_update import enif_update
from ert.config import GenDataConfig, GenKwConfig, SurfaceConfig
from ert.config.gen_kw_config import DataSource
from ert.storage import open_storage

ENSEMBLE_SIZE = 20
NUM_OBS = 5
SURFACE_CONFIG = SurfaceConfig(
    name="TOP_SURFACE",
    forward_init=True,
    update=True,
    ncol=20,
    nrow=25,
    xori=0.0,
    yori=0.0,
    xinc=25.0,
    yinc=25.0,
    rotation=0.0,
    yflip=1,
    forward_init_file="init_surf",
    output_file="out_surf",
    base_surface_path="base_surf",
)
OBSERVATIONS = [
    {
        "type": "general_observation",
        "name": f"OBS_{i}",
        "data": "RESPONSE",
        "restart": 0,
        "index": i,
        "value": 1.0,
        "error": 0.1,
    }
    for i in range(NUM_OBS)
]


def _make_genkw(name: str) -> dict[str, Any]:
    return GenKwConfig(
        name=name,
        group=name,
        distribution={"name": "uniform", "min": 0.8, "max": 1.2},
    ).model_dump(mode="json")


def _populate_prior(prior, rng, genkw_names):
    ncol, nrow = SURFACE_CONFIG.ncol, SURFACE_CONFIG.nrow
    inactive = rng.random((ncol, nrow)) < 0.55

    for real in range(ENSEMBLE_SIZE):
        vals = rng.normal(1.0, 0.1, (ncol, nrow)).astype(np.float32)
        vals[inactive] = np.nan
        prior.save_parameters(
            xr.Dataset({"values": (["x", "y"], vals)}), "TOP_SURFACE", real
        )

    for name in genkw_names:
        prior.save_parameters(
            dataset=pl.DataFrame(
                {
                    name: rng.uniform(0.8, 1.2, ENSEMBLE_SIZE).tolist(),
                    "realization": range(ENSEMBLE_SIZE),
                }
            )
        )

    for iens in range(ENSEMBLE_SIZE):
        prior.save_response(
            "gen_data",
            pl.DataFrame(
                {
                    "response_key": "RESPONSE",
                    "report_step": pl.Series(
                        np.zeros(NUM_OBS, dtype=int), dtype=pl.UInt16
                    ),
                    "index": pl.Series(np.arange(NUM_OBS), dtype=pl.UInt16),
                    "values": pl.Series(rng.normal(1, 0.1, NUM_OBS), dtype=pl.Float32),
                }
            ),
            iens,
        )


@pytest.mark.filterwarnings("ignore::scipy.linalg.LinAlgWarning")
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("param_order", ["genkw_first", "surface_first"])
def test_that_enif_nan_filtering_does_not_contaminate_genkw(tmp_path, param_order):
    rng = np.random.default_rng(42)
    surf_cfg = SURFACE_CONFIG.model_dump(mode="json")

    if param_order == "genkw_first":
        params = [_make_genkw("MULT_1"), _make_genkw("MULT_2"), surf_cfg]
        genkw_names = ["MULT_1", "MULT_2"]
    else:
        params = [surf_cfg, _make_genkw("MULT_2"), _make_genkw("MULT_1")]
        genkw_names = ["MULT_2", "MULT_1"]

    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(
            name="enif_nan_test",
            experiment_config={
                "parameter_configuration": params,
                "response_configuration": [
                    GenDataConfig(keys=["RESPONSE"]).model_dump(mode="json")
                ],
                "observations": OBSERVATIONS,
            },
        )
        prior = storage.create_ensemble(
            experiment, ensemble_size=ENSEMBLE_SIZE, iteration=0, name="prior"
        )
        _populate_prior(prior, rng, genkw_names)

        posterior = storage.create_ensemble(
            experiment,
            ensemble_size=ENSEMBLE_SIZE,
            iteration=1,
            name="posterior",
            prior_ensemble=prior,
        )
        enif_update(
            prior,
            posterior,
            observations=experiment.observation_keys,
            random_seed=42,
        )

        iens = np.arange(ENSEMBLE_SIZE)
        for name in genkw_names:
            post = posterior.load_parameters_numpy(name, iens)
            assert np.all(np.isfinite(post)), f"{name} contains NaN"

        prior_surf = prior.load_parameters_numpy("TOP_SURFACE", iens)
        post_surf = posterior.load_parameters_numpy("TOP_SURFACE", iens)
        nan_mask = np.isnan(prior_surf)

        assert np.array_equal(np.isnan(post_surf), nan_mask)
        assert np.all(np.isfinite(post_surf[~nan_mask]))
        assert not np.array_equal(post_surf[~nan_mask], prior_surf[~nan_mask])


def _make_genkw_no_update(name: str) -> dict[str, Any]:
    return GenKwConfig(
        name=name,
        group=name,
        update=False,
        distribution={"name": "uniform", "min": 0.8, "max": 1.2},
    ).model_dump(mode="json")


def _populate_prior_with_correlated_responses(prior, rng, genkw_names):
    """Populate prior with parameter-dependent responses so EnIF can learn
    a meaningful H map."""
    param_values = {}
    for name in genkw_names:
        vals = rng.uniform(0.8, 1.2, ENSEMBLE_SIZE).tolist()
        param_values[name] = vals
        prior.save_parameters(
            dataset=pl.DataFrame(
                {
                    name: vals,
                    "realization": range(ENSEMBLE_SIZE),
                }
            )
        )

    for iens in range(ENSEMBLE_SIZE):
        # Responses are a linear function of the parameters plus noise
        base = sum(param_values[n][iens] for n in genkw_names) / len(genkw_names)
        response_vals = [base + 0.1 * i + rng.normal(0, 0.01) for i in range(NUM_OBS)]
        prior.save_response(
            "gen_data",
            pl.DataFrame(
                {
                    "response_key": "RESPONSE",
                    "report_step": pl.Series(
                        np.zeros(NUM_OBS, dtype=int), dtype=pl.UInt16
                    ),
                    "index": pl.Series(np.arange(NUM_OBS), dtype=pl.UInt16),
                    "values": pl.Series(response_vals, dtype=pl.Float32),
                }
            ),
            iens,
        )


@pytest.mark.filterwarnings("ignore::scipy.linalg.LinAlgWarning")
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_that_enif_non_updateable_params_are_included_in_h_map_but_not_modified(
    tmp_path,
):
    """Non-updateable numeric params should inform the H map but remain
    unchanged in the posterior."""
    rng = np.random.default_rng(99)

    params = [
        _make_genkw("UPDATABLE_1"),
        _make_genkw("UPDATABLE_2"),
        _make_genkw_no_update("FIXED_1"),
    ]

    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(
            name="enif_non_update_test",
            experiment_config={
                "parameter_configuration": params,
                "response_configuration": [
                    GenDataConfig(keys=["RESPONSE"]).model_dump(mode="json")
                ],
                "observations": OBSERVATIONS,
            },
        )
        prior = storage.create_ensemble(
            experiment, ensemble_size=ENSEMBLE_SIZE, iteration=0, name="prior"
        )
        _populate_prior_with_correlated_responses(
            prior, rng, ["UPDATABLE_1", "UPDATABLE_2", "FIXED_1"]
        )

        posterior = storage.create_ensemble(
            experiment,
            ensemble_size=ENSEMBLE_SIZE,
            iteration=1,
            name="posterior",
            prior_ensemble=prior,
        )
        enif_update(
            prior,
            posterior,
            observations=experiment.observation_keys,
            random_seed=42,
        )

        iens = np.arange(ENSEMBLE_SIZE)

        # Non-updateable param should be unchanged
        prior_fixed = prior.load_parameters_numpy("FIXED_1", iens)
        post_fixed = posterior.load_parameters_numpy("FIXED_1", iens)
        np.testing.assert_array_equal(prior_fixed, post_fixed)

        # At least one updateable param should be modified
        any_updated = False
        for name in ["UPDATABLE_1", "UPDATABLE_2"]:
            prior_arr = prior.load_parameters_numpy(name, iens)
            post_arr = posterior.load_parameters_numpy(name, iens)
            if not np.array_equal(prior_arr, post_arr):
                any_updated = True
        assert any_updated, "At least one updateable param should have been updated"


@pytest.mark.filterwarnings("ignore::scipy.linalg.LinAlgWarning")
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_that_enif_regression_includes_non_updateable_numeric_params(tmp_path):
    """The H map regression should receive all numeric parameters as
    predictors, not just the updateable ones.  Before this change, only
    update=True params were passed to linear_boost_ic_regression, so U
    had fewer columns.

    We spy on linear_boost_ic_regression and verify:
    1. U has columns for both the updateable and non-updateable param.
    2. The posterior of the updateable param is closer to truth than
       the prior (the non-updateable param helps explain the response).
    """
    rng = np.random.default_rng(42)

    n_ens = 50
    n_obs = 8
    slope_true = 1.0
    offset_true = 2.0

    slope_cfg = GenKwConfig(
        name="SLOPE",
        group="SLOPE",
        distribution={"name": "uniform", "min": 0.5, "max": 1.5},
        update=True,
    ).model_dump(mode="json")

    offset_cfg = GenKwConfig(
        name="OFFSET",
        group="DESIGN_MATRIX",
        distribution={"name": "raw"},
        update=False,
        input_source=DataSource.DESIGN_MATRIX,
    ).model_dump(mode="json")

    observations = [
        {
            "type": "general_observation",
            "name": f"OBS_{i}",
            "data": "RESPONSE",
            "restart": 0,
            "index": i,
            "value": slope_true * (i + 1) + offset_true,
            "error": 0.05,
        }
        for i in range(n_obs)
    ]

    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(
            name="enif_regression_test",
            experiment_config={
                "parameter_configuration": [slope_cfg, offset_cfg],
                "response_configuration": [
                    GenDataConfig(keys=["RESPONSE"]).model_dump(mode="json")
                ],
                "observations": observations,
            },
        )
        prior = storage.create_ensemble(
            experiment, ensemble_size=n_ens, iteration=0, name="prior"
        )

        slope_vals = rng.uniform(0.5, 1.5, n_ens)
        offset_vals = rng.uniform(0.5, 3.5, n_ens)

        prior.save_parameters(
            pl.DataFrame({"SLOPE": slope_vals.tolist(), "realization": range(n_ens)})
        )
        prior.save_parameters(
            pl.DataFrame({"OFFSET": offset_vals.tolist(), "realization": range(n_ens)})
        )

        for iens in range(n_ens):
            response_vals = [
                slope_vals[iens] * (i + 1) + offset_vals[iens] + rng.normal(0, 0.01)
                for i in range(n_obs)
            ]
            prior.save_response(
                "gen_data",
                pl.DataFrame(
                    {
                        "response_key": "RESPONSE",
                        "report_step": pl.Series(
                            np.zeros(n_obs, dtype=int), dtype=pl.UInt16
                        ),
                        "index": pl.Series(np.arange(n_obs), dtype=pl.UInt16),
                        "values": pl.Series(response_vals, dtype=pl.Float32),
                    }
                ),
                iens,
            )

        posterior = storage.create_ensemble(
            experiment,
            ensemble_size=n_ens,
            iteration=1,
            name="posterior",
            prior_ensemble=prior,
        )

        captured_U = {}

        def spy_regression(U, Y, **kwargs):
            captured_U["U"] = U.copy()
            return linear_boost_ic_regression(U, Y, **kwargs)

        with patch(
            "ert.analysis._enif_update.linear_boost_ic_regression",
            side_effect=spy_regression,
        ):
            enif_update(
                prior,
                posterior,
                observations=experiment.observation_keys,
                random_seed=42,
            )

        # 1) The regression received both params as predictors.
        #    Before our change U would have had only 1 column (SLOPE).
        assert captured_U["U"].shape[1] == 2, (
            f"Expected 2 predictor columns (SLOPE + OFFSET), "
            f"got {captured_U['U'].shape[1]}"
        )

        iens_all = np.arange(n_ens)

        # 2) OFFSET (non-updateable) must be unchanged.
        np.testing.assert_array_equal(
            prior.load_parameters_numpy("OFFSET", iens_all),
            posterior.load_parameters_numpy("OFFSET", iens_all),
        )

        # 3) With OFFSET in the H map, EnIF can separate the two effects
        #    and the posterior variance of SLOPE collapses dramatically
        #    (std ~0.008 vs prior std ~0.28).  Without OFFSET in the H map
        #    (old behavior) the posterior std stays around 0.13.
        prior_slope = prior.load_parameters_numpy("SLOPE", iens_all).flatten()
        post_slope = posterior.load_parameters_numpy("SLOPE", iens_all).flatten()

        variance_reduction = 1 - post_slope.var() / prior_slope.var()
        assert variance_reduction > 0.95, (
            f"Expected >95% variance reduction when OFFSET is in the H map, "
            f"got {variance_reduction:.1%}"
        )
