import json
import pprint

import pytest
from ert_shared.storage.blob_api import BlobApi
from ert_shared.storage.rdb_api import RdbApi
from ert_shared.storage.storage_api import StorageApi

from tests.storage import db_info


def test_response(db_info):
    populated_db, db_lookup = db_info
    with StorageApi(rdb_url=populated_db, blob_url=populated_db) as api:
        schema = api.get_response(db_lookup["ensemble"], "response_one", None)
        assert len(schema["realizations"]) == 2
        assert len(schema["observations"]) == 1
        assert len(schema["observations"][0]["data"]) == 5


def test_ensembles(db_info):
    populated_db, _ = db_info
    with StorageApi(rdb_url=populated_db, blob_url=populated_db) as api:
        schema = api.get_ensembles()


def test_ensemble(db_info):
    populated_db, db_lookup = db_info
    with StorageApi(rdb_url=populated_db, blob_url=populated_db) as api:
        schema = api.get_ensemble(db_lookup["ensemble"])


def test_realization(db_info):
    populated_db, db_lookup = db_info
    with StorageApi(rdb_url=populated_db, blob_url=populated_db) as api:
        schema = api.get_realization(
            ensemble_id=db_lookup["ensemble"], realization_idx=0, filter=None
        )


def test_priors(db_info):
    populated_db, db_lookup = db_info
    with StorageApi(rdb_url=populated_db, blob_url=populated_db) as api:
        schema = api.get_ensemble(db_lookup["ensemble"])
        assert {
            "group": "group",
            "key": "key1",
            "prior": {
                "function": "function",
                "parameter_names": ["paramA", "paramB"],
                "parameter_values": [0.1, 0.2],
            },
            "parameter_ref": 3,
        } in schema["parameters"]


def test_parameter(db_info):
    populated_db, db_lookup = db_info
    with StorageApi(rdb_url=populated_db, blob_url=populated_db) as api:
        ens_id = db_lookup["ensemble"]
        par_def_id = db_lookup["parameter_def_key1_group"]
        schema = api.get_parameter(ensemble_id=ens_id, parameter_def_id=par_def_id)
        assert schema["key"] == "key1"
        assert schema["group"] == "group"
        assert schema["prior"]["function"] == "function"


def test_nonexisiting_parameter(db_info):
    populated_db, db_lookup = db_info
    with StorageApi(rdb_url=populated_db, blob_url=populated_db) as api:
        schema = api.get_parameter(
            ensemble_id=db_lookup["ensemble"], parameter_def_id="1293495"
        )
        assert schema is None


def test_observation(db_info):
    populated_db, _ = db_info
    name = "observation_one"
    with StorageApi(rdb_url=populated_db, blob_url=populated_db) as api:
        obs = api.get_observation(name)
        assert obs == {
            "attributes": {"region": "1"},
            "name": name,
            "data": {
                "data_indexes": {"data_ref": 2},
                "key_indexes": {"data_ref": 1},
                "std": {"data_ref": 4},
                "values": {"data_ref": 3},
            },
        }


def test_observation_attributes(db_info):
    populated_db, _ = db_info
    attr = "region"
    value = "1"
    name = "observation_one"
    expected = {"attributes": {attr: value}}

    with StorageApi(rdb_url=populated_db, blob_url=populated_db) as api:
        api.set_observation_attribute(name, attr, value)

    with StorageApi(rdb_url=populated_db, blob_url=populated_db) as api:
        assert api.get_observation_attribute(name, attr) == expected


def test_single_observation_misfit_calculation(db_info):
    populated_db, _ = db_info
    # observation
    values_obs = [10.1, 10.2]
    stds_obs = [1, 3]
    data_indexes_obs = [2, 3]
    # response
    values_res = [11.1, 11.2, 9.9, 9.3]

    misfit_expected = {
        "observation_one": [
            {
                "value": ((values_res[index] - obs_value) / obs_std) ** 2,
                "sign": values_res[index] - obs_value > 0,
                "obs_index": obs_index,
            }
            for obs_index, (obs_value, obs_std, index) in enumerate(
                zip(values_obs, stds_obs, data_indexes_obs)
            )
        ]
    }

    with StorageApi(rdb_url=populated_db, blob_url=populated_db) as api:
        univariate_misfit = api.get_response(
            ensemble_id=1, response_name="response_one", filter=None
        )

        assert (
            univariate_misfit["realizations"][0]["univariate_misfits"]
            == misfit_expected
        )
