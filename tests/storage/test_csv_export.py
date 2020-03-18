import ert_shared.storage.storage_api as storage_api
import pandas as pd
import pytest
from ert_shared.storage.repository import ErtRepository

from ert_shared.storage.extraction_api import (
    _dump_observations,
    _dump_parameters,
    _dump_response,
)

from tests.storage import db_session, engine, tables

observation_data = {
    ("POLY_OBS", 0, 10): {"OBS": 2.0, "STD": 0.1},
    ("POLY_OBS", 2, 12): {"OBS": 7.1, "STD": 1.1},
    ("POLY_OBS", 4, 14): {"OBS": 21.1, "STD": 4.1},
    ("POLY_OBS", 6, 16): {"OBS": 31.8, "STD": 9.1},
    ("POLY_OBS", 8, 18): {"OBS": 53.2, "STD": 16.1},
    ("TEST_OBS", 3, 3): {"OBS": 6, "STD": 0.1},
    ("TEST_OBS", 6, 6): {"OBS": 12, "STD": 0.2},
    ("TEST_OBS", 9, 9): {"OBS": 18, "STD": 0.3},
}

poly_res = pd.DataFrame.from_dict(
    {
        0: {
            0: 2.5995,
            1: 5.203511,
            2: 9.496884000000001,
            3: 15.479619,
            4: 23.151716,
            5: 32.513175000000004,
            6: 43.563995999999996,
            7: 56.304179,
            8: 70.73372400000001,
            9: 86.852631,
        },
        1: {
            0: 4.97204,
            1: 6.23818,
            2: 8.18051,
            3: 10.79903,
            4: 14.09374,
            5: 18.06464,
            6: 22.71173,
            7: 28.035009999999996,
            8: 34.03448,
            9: 40.71014,
        },
        2: {
            0: 0.660302,
            1: 1.906593,
            2: 4.597682,
            3: 8.733569000000001,
            4: 14.314254,
            5: 21.339737000000003,
            6: 29.810018000000003,
            7: 39.725097000000005,
            8: 51.084974,
            9: 63.889649000000006,
        },
        3: {
            0: 4.99478,
            1: 5.566743000000001,
            2: 6.4375800000000005,
            3: 7.607291,
            4: 9.075876000000001,
            5: 10.843335,
            6: 12.909668,
            7: 15.274875,
            8: 17.938955999999997,
            9: 20.901911,
        },
        4: {
            0: 1.96728,
            1: 2.2754027,
            2: 3.0749214,
            3: 4.3658361,
            4: 6.1481468,
            5: 8.421853500000001,
            6: 11.186956200000001,
            7: 14.4434549,
            8: 18.1913496,
            9: 22.430640299999997,
        },
    }
)

responses = {"POLY_RES": poly_res}


def test_retrieve_response_data(db_session):
    ensemble_name = "default"
    with ErtRepository(db_session) as repository:
        ensemble = repository.add_ensemble(name=ensemble_name)

        observations = pd.DataFrame.from_dict(observation_data)
        _dump_observations(repository=repository, observations=observations)

        for i in range(5):
            repository.add_realization(i, ensemble.name)

        key_mapping = {"POLY_RES": "POLY_OBS"}

        _dump_response(
            repository=repository,
            responses=responses,
            ensemble_name=ensemble.name,
            key_mapping=key_mapping,
        )
        repository.commit()

    with ErtRepository(db_session) as repository:
        for response in storage_api.get_response_data("POLY_RES", ensemble_name, repository=repository):
            realization_values = poly_res[response.index].to_list()
            assert response.values == realization_values
