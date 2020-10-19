import os

import pytest
from ert_shared.storage import ERT_STORAGE
from ert_shared.storage.blob_api import BlobApi
from ert_shared.storage.blobs_model import Blobs
from ert_shared.storage.entities_model import Entities
from ert_shared.storage.rdb_api import RdbApi
from ert_shared.storage.storage_api import StorageApi
from sqlalchemy import create_engine
from sqlalchemy.orm import Session


@pytest.fixture(scope="session")
def initialize_databases(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("database")

    rdb_url = f"sqlite:///{tmp_path}/entities.db"
    blob_url = f"sqlite:///{tmp_path}/blobs.db"

    ERT_STORAGE.initialize(rdb_url=rdb_url, blob_url=blob_url)


@pytest.fixture(scope="module")
def populated_database(initialize_databases):
    rdb_session = ERT_STORAGE.RdbSession()
    blob_session = ERT_STORAGE.BlobSession()

    rdb_api = RdbApi(session=rdb_session)
    blob_api = BlobApi(session=blob_session)

    def add_blob(data):
        ret = blob_api.add_blob(data)
        return ret.id

    db_lookup = {}
    ######## add priors ########
    prior_key1 = rdb_api.add_prior(
        "group", "key1", "function", ["paramA", "paramB"], [0.1, 0.2]
    )
    prior_key2 = rdb_api.add_prior(
        "group", "key2", "function", ["paramA", "paramB"], [0.3, 0.4]
    )
    prior_key3 = rdb_api.add_prior(
        "group", "key3", "function", ["paramA", "paramB"], [0.5, 0.6]
    )
    prior_A = rdb_api.add_prior("G", "A", "function", ["paramA", "paramB"], [0.5, 0.6])

    ######## add ensemble ########
    ensemble = rdb_api.add_ensemble(
        name="ensemble_name", priors=[prior_key1, prior_key2, prior_key3]
    )
    db_lookup["ensemble"] = ensemble.id
    db_lookup["ensemble_timestamp"] = ensemble.time_created
    ######## add parameteredefinitionss ########
    parameter_def_A_G = rdb_api.add_parameter_definition(
        "A", "G", "ensemble_name", prior=prior_A
    )
    parameter_def_B_G = rdb_api.add_parameter_definition("B", "G", "ensemble_name")
    parameter_def_key1_group = rdb_api.add_parameter_definition(
        "key1", "group", "ensemble_name", prior=prior_key1
    )
    db_lookup["parameter_def_A_G"] = parameter_def_A_G.id
    db_lookup["parameter_def_key1_group"] = parameter_def_key1_group.id

    ######## add observations ########
    observation_one = rdb_api.add_observation(
        name="observation_one",
        key_indexes_ref=add_blob([0, 3]),
        data_indexes_ref=add_blob([2, 3]),
        values_ref=add_blob([10.1, 10.2]),
        stds_ref=add_blob([1, 3]),
    )
    observation_one.add_attribute("region", "1")

    observation_two_first = rdb_api.add_observation(
        name="observation_two_first",
        key_indexes_ref=add_blob(["2000-01-01 20:01:01"]),
        data_indexes_ref=add_blob([4]),
        values_ref=add_blob([10.3]),
        stds_ref=add_blob([2]),
    )

    observation_two_second = rdb_api.add_observation(
        name="observation_two_second",
        key_indexes_ref=add_blob(["2000-01-02 20:01:01"]),
        data_indexes_ref=add_blob([5]),
        values_ref=add_blob([10.4]),
        stds_ref=add_blob([2.5]),
    )

    ######## add response definitions ########
    response_definition_one = rdb_api.add_response_definition(
        name="response_one",
        indexes_ref=add_blob([3, 5, 8, 9]),
        ensemble_name=ensemble.name,
    )

    response_definition_two = rdb_api.add_response_definition(
        name="response_two",
        indexes_ref=add_blob(
            [
                "2000-01-01 20:01:01",
                "2000-01-02 20:01:01",
                "2000-01-02 20:01:01",
                "2000-01-02 20:01:01",
                "2000-01-02 20:01:01",
                "2000-01-02 20:01:01",
            ]
        ),
        ensemble_name=ensemble.name,
    )
    db_lookup["response_defition_one"] = response_definition_one.id

    ######## observation response definition links ########
    obs_res_def_link = rdb_api._add_observation_response_definition_link(
        observation_id=observation_one.id,
        response_definition_id=response_definition_one.id,
        active_ref=add_blob([True, False]),
        update_id=None,
    )

    rdb_api._add_observation_response_definition_link(
        observation_id=observation_two_first.id,
        response_definition_id=response_definition_two.id,
        active_ref=add_blob([True]),
        update_id=None,
    )

    rdb_api._add_observation_response_definition_link(
        observation_id=observation_two_second.id,
        response_definition_id=response_definition_two.id,
        active_ref=add_blob([True]),
        update_id=None,
    )

    ######## add realizations ########
    realization_0 = rdb_api.add_realization(0, ensemble.name)
    realization_1 = rdb_api.add_realization(1, ensemble.name)
    db_lookup["realization_0"] = realization_0.id

    def add_data(realization, response_def, ens):
        response_one = rdb_api.add_response(
            name="response_one",
            values_ref=add_blob([11.1, 11.2, 9.9, 9.3]),
            realization_index=realization.index,
            ensemble_name=ens.name,
        )
        rdb_api._add_misfit(200, obs_res_def_link.id, response_one.id)

        rdb_api.add_response(
            name="response_two",
            values_ref=add_blob([12.1, 12.2, 11.1, 11.2, 9.9, 9.3]),
            realization_index=realization.index,
            ensemble_name=ens.name,
        )

        rdb_api.add_parameter("A", "G", add_blob(1), realization.index, "ensemble_name")
        rdb_api.add_parameter("B", "G", add_blob(2), realization.index, "ensemble_name")
        rdb_api.add_parameter(
            "key1", "group", add_blob(2), realization.index, "ensemble_name"
        )

    add_data(realization_0, response_def=response_definition_one, ens=ensemble)
    add_data(realization_1, response_def=response_definition_one, ens=ensemble)

    ######## add blob #########
    data_blob = add_blob([0, 1, 2, 3])
    db_lookup["data_blob"] = data_blob

    rdb_session.commit()
    blob_session.commit()

    rdb_session.close()
    blob_session.close()

    yield db_lookup


@pytest.fixture
def apis(initialize_databases):
    rdb_session = ERT_STORAGE.RdbSession()
    blob_session = ERT_STORAGE.BlobSession()

    rdb_api = RdbApi(session=rdb_session)
    blob_api = BlobApi(session=blob_session)

    try:
        yield rdb_api, blob_api
    finally:
        rdb_session.rollback()
        blob_session.rollback()
        rdb_session.close()
        blob_session.close()


@pytest.fixture
def db_apis(populated_database):
    db_lookup = populated_database

    rdb_session = ERT_STORAGE.RdbSession()
    blob_session = ERT_STORAGE.BlobSession()

    rdb_api = RdbApi(session=rdb_session)
    blob_api = BlobApi(session=blob_session)

    try:
        yield rdb_api, blob_api, db_lookup
    finally:
        rdb_session.rollback()
        blob_session.rollback()
        rdb_session.close()
        blob_session.close()


@pytest.fixture
def storage_api(db_apis):
    rdb_api, blob_api, db_lookup = db_apis
    yield StorageApi(rdb_api=rdb_api, blob_api=blob_api), db_lookup
