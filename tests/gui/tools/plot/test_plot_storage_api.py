import pytest
from unittest import TestCase

from ert_shared.storage.storage_api import PlotStorageApi

from tests.storage import populated_db, db_session


def test_all_keys(populated_db):
    api = PlotStorageApi()
    assert api.all_data_type_keys() == ["observation_one", "response_one"]

def test_observation_values():
    api = PlotStorageApi()
    result = api.data_for_key(case="ensemble_name", key="observation_one")
    assert result == ([22.1, 44.2], [1, 3])

def test_parameter_values():
    pass

def test_response_values():
    api = PlotStorageApi()
    result = api.data_for_key(case="ensemble_name", key="response_one")
    assert result == ([22.1, 44.2], [0, 1])