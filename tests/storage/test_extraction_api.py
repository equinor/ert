import pandas as pd
import pytest

from ert_shared.storage.repository import ErtRepository
from ert_shared.storage.extraction_api import _dump_observations

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


def test_dump_observations(db_session):
    with ErtRepository(db_session) as repository:
        observations = pd.DataFrame.from_dict(observation_data)
        _dump_observations(repository=repository, observations=observations)
        repository.commit()

    with ErtRepository(db_session) as repository:
        poly_obs = repository.get_observation("POLY_OBS")
        assert poly_obs.id is not None
        assert poly_obs.key_indexes == [0, 2, 4, 6, 8]
        assert poly_obs.data_indexes == [10, 12, 14, 16, 18]
        assert poly_obs.values == [2.0, 7.1, 21.1, 31.8, 53.2]
        assert poly_obs.stds == [0.1, 1.1, 4.1, 9.1, 16.1]

        test_obs = repository.get_observation("TEST_OBS")
        assert test_obs.id is not None
        assert test_obs.key_indexes == [3, 6, 9]
        assert test_obs.data_indexes == [3, 6, 9]
        assert test_obs.values == [6, 12, 18]
        assert test_obs.stds == [0.1, 0.2, 0.3]
