import logging

import numpy as np
import polars as pl

from ert.analysis import smoother_update
from ert.config import ObservationSettings
from ert.config._observations import RFTObservation
from ert.config.rft_config import RFTConfig
from tests.ert.defaults_generator import (
    create_rft_location_metadata,
    create_rft_observation,
    create_rft_response,
)

_ACTIVE_LOCATION_1 = (100.0, 100.0, 100.0)
_OUTSIDE_GRID_LOCATION = (101.0, 101.0, 101.0)
_ACTIVE_LOCATION_2 = (102.0, 102.0, 102.0)

_ACTIVE_CELL_1 = (1, 1, 1)
_ACTIVE_CELL_2 = (1, 1, 3)


def _rft_responses(iens: int) -> pl.DataFrame:
    # The value must vary across realizations, otherwise the observation
    # collapses on `std_cutoff` and is deactivated for that reason instead.
    return pl.concat(
        create_rft_response(i=i, j=j, k=k, value=148.0 + iens)
        for i, j, k in [
            _ACTIVE_CELL_1,
            _ACTIVE_CELL_2,
        ]
    )


def _rft_location_metadata() -> pl.DataFrame:
    """Maps the observation locations to grid cells, where a null
    `well_connection_cell` means the location is outside the grid.
    """
    return pl.concat(
        create_rft_location_metadata(
            east=east,
            north=north,
            tvd=tvd,
            well_connection_cell=connection_cell,
        )
        for connection_cell, (east, north, tvd) in [
            (_ACTIVE_CELL_1, _ACTIVE_LOCATION_1),
            (None, _OUTSIDE_GRID_LOCATION),
            (_ACTIVE_CELL_2, _ACTIVE_LOCATION_2),
        ]
    )


def _rft_observations() -> list[RFTObservation]:
    return [
        create_rft_observation(
            name=name,
            east=east,
            north=north,
            tvd=tvd,
        )
        for name, (east, north, tvd) in [
            ("RFT_OBS_ACTIVE_1", _ACTIVE_LOCATION_1),
            ("RFT_OBS_OUTSIDE_GRID", _OUTSIDE_GRID_LOCATION),
            ("RFT_OBS_ACTIVE_2", _ACTIVE_LOCATION_2),
        ]
    ]


def _create_rft_ensembles(
    storage, ensemble_size: int, rft_observations: list[RFTObservation]
):
    experiment = storage.create_experiment(
        name="rft_smoother_update",
        experiment_config={
            "response_configuration": [
                RFTConfig(input_files=["BASE.RFT"]).model_dump(mode="json")
            ],
            "observations": [o.model_dump(mode="json") for o in rft_observations],
        },
    )
    prior = storage.create_ensemble(
        experiment, ensemble_size=ensemble_size, iteration=0, name="prior"
    )
    for iens in range(ensemble_size):
        prior.save_response("rft", _rft_responses(iens), iens)
        prior.save_observation_location_metadata(_rft_location_metadata(), iens)

    posterior = storage.create_ensemble(
        experiment,
        ensemble_size=ensemble_size,
        iteration=1,
        name="posterior",
        prior_ensemble=prior,
    )
    return prior, posterior


def test_that_smoother_update_logs_how_many_rft_observations_were_outside_the_grid(
    storage, caplog
):
    prior, posterior = _create_rft_ensembles(
        storage, ensemble_size=3, rft_observations=_rft_observations()
    )

    with caplog.at_level(logging.INFO):
        smoother_update(
            prior,
            posterior,
            prior.experiment.observation_keys,
            ObservationSettings(),
            rng=np.random.default_rng(42),
            strategy_map={},
        )

    assert (
        "Update step 0: 1 of 3 RFT observations deactivated because their "
        "location was outside the grid" in caplog.text
    )
