from functools import partial

import numpy as np
import polars as pl
import scipy as sp
import xarray as xr
import xtgeo
from scipy.ndimage import gaussian_filter

from ert.analysis import smoother_update
from ert.config import ESSettings, Field, GenDataConfig, ObservationSettings
from ert.field_utils import Shape


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

    response_config = GenDataConfig(keys=["RESPONSE"])
    obs = [
        {
            "type": "general_observation",
            "name": "OBSERVATION",
            "data": "RESPONSE",
            "restart": 0,
            "index": i,
            "value": float(observations[i]),
            "error": 1.0,
        }
        for i in range(len(observations))
    ]

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
            "parameter_configuration": [config.model_dump(mode="json")],
            "response_configuration": [response_config.model_dump(mode="json")],
            "observations": obs,
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
