import json

import pytest
from ert_shared.storage.blob_api import BlobApi
from ert_shared.storage.rdb_api import RdbApi
from tests.storage import (
    apis,
    db_apis,
    populated_database,
    initialize_databases,
    storage_api,
)


def test_response(storage_api):
    api, db_lookup = storage_api
    schema = api.get_response(db_lookup["ensemble"], "response_one", None)
    assert len(schema["realizations"]) == 2
    assert len(schema["observations"]) == 1
    assert len(schema["observations"][0]["data"]) == 5

    schema = api.get_response(db_lookup["ensemble"], "response_not_existing", None)
    assert schema is None


def test_ensembles(storage_api):
    api, db_lookup = storage_api
    schema = api.get_ensembles()
    assert type(schema["ensembles"]) == list
    assert {
        "name": "ensemble_name",
        "time_created": db_lookup["ensemble_timestamp"].isoformat(),
        "parent": {},
        "children": [],
        "ensemble_ref": db_lookup["ensemble"],
    } in schema["ensembles"]


def test_ensemble(storage_api):
    api, db_lookup = storage_api
    schema = api.get_ensemble(db_lookup["ensemble"])
    assert schema["name"] == "ensemble_name"
    assert schema["time_created"] == db_lookup["ensemble_timestamp"].isoformat()
    assert schema["parent"] == {}
    assert schema["children"] == []
    assert schema["ensemble_ref"] == db_lookup["ensemble"]
    assert {
        "group": "G",
        "key": "A",
        "parameter_ref": db_lookup["parameter_def_A_G"],
        "prior": {
            "function": "function",
            "parameter_names": ["paramA", "paramB"],
            "parameter_values": [0.5, 0.6],
        },
    } in schema["parameters"]
    assert {"name": 0, "realization_ref": 0} in schema["realizations"]
    assert {"name": "response_one", "response_ref": "response_one"} in schema[
        "responses"
    ]


def test_realization(storage_api):
    api, db_lookup = storage_api
    schema = api.get_realization(
        ensemble_id=db_lookup["ensemble"], realization_idx=0, filter=None
    )
    assert schema["name"] == 0
    assert len(schema["responses"]) == 2
    assert len(schema["parameters"]) == 3


def test_priors(storage_api):
    api, db_lookup = storage_api
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


def test_parameter(storage_api):
    api, db_lookup = storage_api
    ens_id = db_lookup["ensemble"]
    par_def_id = db_lookup["parameter_def_key1_group"]
    schema = api.get_parameter(ensemble_id=ens_id, parameter_def_id=par_def_id)
    assert schema["key"] == "key1"
    assert schema["group"] == "group"
    assert schema["prior"]["function"] == "function"


def test_nonexisiting_parameter(storage_api):
    api, db_lookup = storage_api
    schema = api.get_parameter(
        ensemble_id=db_lookup["ensemble"], parameter_def_id="1293495"
    )
    assert schema is None


def test_observation(storage_api):
    api, _ = storage_api
    name = "observation_one"
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


def test_observation_attributes(storage_api):
    api, _ = storage_api
    attr = "region"
    value = "1"
    name = "observation_one"
    expected = {"attributes": {attr: value}}

    api.set_observation_attribute(name, attr, value)
    assert api.get_observation_attribute(name, attr) == expected


def test_single_observation_misfit_calculation(storage_api):
    api, _ = storage_api
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

    univariate_misfit = api.get_response(
        ensemble_id=1, response_name="response_one", filter=None
    )

    assert univariate_misfit["realizations"][0]["univariate_misfits"] == misfit_expected


def test_data(storage_api):
    api, db_lookup = storage_api
    blob = api.get_data(db_lookup["data_blob"])
    assert blob is not None
    blob = api.get_data("non_existing")
    assert blob is None
