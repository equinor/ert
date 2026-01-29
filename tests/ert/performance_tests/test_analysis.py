import datetime as _datetime
from functools import partial

# Inlined observation helpers (originally in ert.storage.observation_helpers)
from typing import Any

import numpy as np
import polars as pl
import scipy as sp
import xarray as xr
import xtgeo
from scipy.ndimage import gaussian_filter

from ert.analysis import smoother_update
from ert.config import ESSettings, Field, GenDataConfig, ObservationSettings
from ert.field_utils import Shape


def _to_iso_date(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "isoformat"):
        try:
            return value.date().isoformat()
        except Exception:
            return value.isoformat()
    try:
        import pandas as pd

        if isinstance(value, pd.Timestamp):
            return value.date().isoformat()
    except Exception:
        pass
    try:
        ms = int(value)
        dt = _datetime.datetime.fromtimestamp(ms / 1000.0)
        return dt.date().isoformat()
    except Exception:
        return str(value)


def dataframe_to_declarations(
    group_name: str, df: pl.DataFrame
) -> list[dict[str, Any]]:
    decls: list[dict[str, Any]] = []
    for row in df.iter_rows(named=True):
        row = dict(row)
        if "time" in row and "response_key" in row:
            date = _to_iso_date(row.get("time"))
            decls.append(
                {
                    "type": "summary_observation",
                    "name": row.get("observation_key") or group_name,
                    "key": row.get("response_key"),
                    "date": date,
                    "value": float(row.get("observations")),
                    "error": float(row.get("std")),
                    "east": None if row.get("east") is None else float(row.get("east")),
                    "north": None
                    if row.get("north") is None
                    else float(row.get("north")),
                    "radius": None
                    if row.get("radius") is None
                    else float(row.get("radius")),
                }
            )
            continue
        if "index" in row or "report_step" in row:
            decls.append(
                {
                    "type": "general_observation",
                    "name": row.get("observation_key") or group_name,
                    "data": row.get("response_key"),
                    "value": float(row.get("observations")),
                    "error": float(row.get("std")),
                    "restart": int(row.get("report_step", 0) or 0),
                    "index": int(row.get("index", 0) or 0),
                    "east": None if row.get("east") is None else float(row.get("east")),
                    "north": None
                    if row.get("north") is None
                    else float(row.get("north")),
                    "radius": None
                    if row.get("radius") is None
                    else float(row.get("radius")),
                }
            )
            continue
        if (
            "tvd" in row
            or "well" in row
            or ("response_key" in row and ":" in str(row.get("response_key", "")))
        ):
            resp = str(row.get("response_key", ""))
            try:
                well, date_str, prop = resp.split(":", 2)
            except Exception:
                well = row.get("well") or group_name
                date_str = _to_iso_date(row.get("time") or row.get("date"))
                prop = row.get("property") or "PRESSURE"
            decls.append(
                {
                    "type": "rft_observation",
                    "name": row.get("observation_key") or group_name,
                    "well": well,
                    "date": date_str,
                    "property": prop,
                    "value": float(row.get("observations")),
                    "error": float(row.get("std")),
                    "north": None
                    if row.get("north") is None
                    else float(row.get("north")),
                    "east": None if row.get("east") is None else float(row.get("east")),
                    "tvd": None if row.get("tvd") is None else float(row.get("tvd")),
                }
            )
            continue
        decls.append(
            {
                "type": "general_observation",
                "name": group_name,
                "data": row.get("response_key") or "",
                "value": float(row.get("observations", 0.0) or 0.0),
                "error": float(row.get("std", 1.0) or 1.0),
                "restart": int(row.get("report_step", 0) or 0),
                "index": int(row.get("index", 0) or 0),
            }
        )
    return decls


def dataframes_to_declarations(dfs: dict[str, pl.DataFrame]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name, df in dfs.items():
        out.extend(dataframe_to_declarations(name, df))
    return out


def test_and_benchmark_adaptive_localization_with_fields(
    storage, tmp_path, monkeypatch, benchmark
):
    monkeypatch.chdir(tmp_path)

    rng = np.random.default_rng(42)

    num_grid_cells = 500
    num_parameters = num_grid_cells * num_grid_cells
    num_observations = 50
    num_ensemble = 25

    # Create a tridiagonal matrix that maps responses to parameters.
    # Being tridiagonal, it ensures that each response is influenced
    # only by its neighboring parameters.
    diagonal = np.ones(min(num_parameters, num_observations))
    A = sp.sparse.diags(
        [diagonal, diagonal, diagonal],
        offsets=[-1, 0, 1],
        shape=(num_observations, num_parameters),
        dtype=float,
    ).toarray()

    # We add some noise that is insignificant compared to the
    # actual local structure in the forward model step
    A += rng.standard_normal(size=A.shape) * 0.01

    def g(X):
        """Apply the forward model."""
        return A @ X

    all_realizations = np.zeros((num_ensemble, num_grid_cells, num_grid_cells, 1))

    # Generate num_ensemble realizations of the Gaussian Random Field
    for i in range(num_ensemble):
        sigma = 10
        realization = np.exp(
            gaussian_filter(
                gaussian_filter(
                    rng.standard_normal((num_grid_cells, num_grid_cells)), sigma=sigma
                ),
                sigma=sigma,
            )
        )

        realization = realization[..., np.newaxis]
        all_realizations[i] = realization

    X = all_realizations.reshape(-1, num_grid_cells * num_grid_cells).T

    Y = g(X)

    # Create observations by adding noise to a realization.
    observation_noise = rng.standard_normal(size=num_observations)
    observations = Y[:, 0] + observation_noise

    # Create necessary files and data sets to be able to update
    # the parameters using the ensemble smoother.
    shape = Shape(num_grid_cells, num_grid_cells, 1)
    grid = xtgeo.create_box_grid(dimension=(shape.nx, shape.ny, shape.nz))
    grid.to_file("MY_EGRID.EGRID", "egrid")

    resp = GenDataConfig(keys=["RESPONSE"])
    obs = pl.DataFrame(
        {
            "response_key": "RESPONSE",
            "observation_key": "OBSERVATION",
            "report_step": 0,
            "index": np.arange(len(observations)),
            "observations": observations,
            "std": observation_noise,
            "east": pl.Series([None] * len(observations), dtype=pl.Float32),
            "north": pl.Series([None] * len(observations), dtype=pl.Float32),
            "radius": pl.Series([None] * len(observations), dtype=pl.Float32),
        }
    )

    param_group = "PARAM_FIELD"

    config = Field.from_config_list(
        "MY_EGRID.EGRID",
        [
            param_group,
            param_group,
            "param.GRDECL",
            {
                "INIT_FILES": "param_%d.GRDECL",
                "FORWARD_INIT": "False",
            },
        ],
    )

    experiment = storage.create_experiment(
        experiment_config={
            "parameter_configuration": [config],
            "response_configuration": [resp],
            "observations": dataframes_to_declarations({"gen_data": obs}),
        }
    )

    prior_ensemble = storage.create_ensemble(
        experiment,
        ensemble_size=num_ensemble,
        iteration=0,
        name="prior",
    )

    for iens in range(prior_ensemble.ensemble_size):
        prior_ensemble.save_parameters(
            xr.Dataset(
                {
                    "values": xr.DataArray(
                        X[:, iens].reshape(num_grid_cells, num_grid_cells, 1),
                        dims=("x", "y", "z"),
                    ),
                }
            ),
            param_group,
            iens,
        )

        prior_ensemble.save_response(
            "gen_data",
            pl.DataFrame(
                {
                    "response_key": "RESPONSE",
                    "report_step": 0,
                    "index": range(len(Y[:, iens])),
                    "values": Y[:, iens],
                }
            ),
            iens,
        )

    posterior_ensemble = storage.create_ensemble(
        prior_ensemble.experiment_id,
        ensemble_size=prior_ensemble.ensemble_size,
        iteration=1,
        name="posterior",
        prior_ensemble=prior_ensemble,
    )

    smoother_update_run = partial(
        smoother_update,
        prior_ensemble,
        posterior_ensemble,
        ["OBSERVATION"],
        [param_group],
        ObservationSettings(),
        ESSettings(localization=True),
    )
    benchmark(smoother_update_run)

    prior_da = prior_ensemble.load_parameters(param_group, range(num_ensemble))[
        "values"
    ]
    posterior_da = posterior_ensemble.load_parameters(param_group, range(num_ensemble))[
        "values"
    ]
    # Make sure some, but not all parameters were updated.
    assert not np.allclose(prior_da, posterior_da)
    # All parameters would be updated with a global update so this would fail.
    assert np.isclose(prior_da, posterior_da).sum() > 0
    # The std for the ensemble should decrease
    assert float(
        prior_ensemble.calculate_std_dev_for_parameter_group(param_group).sum()
    ) > float(
        posterior_ensemble.calculate_std_dev_for_parameter_group(param_group).sum()
    )
