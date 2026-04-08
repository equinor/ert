from typing import Any

import numpy as np
import polars as pl
import pytest
import xarray as xr

from ert.analysis._enif_update import enif_update
from ert.config import GenDataConfig, GenKwConfig, SurfaceConfig
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
            parameters=list(experiment.parameter_configuration.keys()),
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
