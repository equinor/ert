import os

import pytest
from ert_shared.storage.blob_api import BlobApi
from ert_shared.storage.model import Blobs, Entities
from ert_shared.storage.rdb_api import RdbApi
from sqlalchemy import create_engine
from sqlalchemy.orm import Session


@pytest.fixture(scope="session")
def engine():
    engine = create_engine("sqlite:///:memory:", echo=False)
    engine.execute("pragma foreign_keys=on")
    return engine


@pytest.fixture(scope="session")
def tables(engine):
    Entities.metadata.create_all(engine)
    Blobs.metadata.create_all(engine)
    yield
    Entities.metadata.drop_all(engine)
    Blobs.metadata.drop_all(engine)


@pytest.fixture
def db_connection(engine, tables):
    """Returns an sqlalchemy session, and after the test tears down everything properly."""
    connection = engine.connect()
    transaction = connection.begin()

    yield connection

    transaction.rollback()
    connection.close()


@pytest.fixture(scope="module")
def db_info(tmp_path_factory):
    def add_blob(data):
        ret = blob.add_blob(data)
        blob.flush()
        return ret.id

    tmp_path = tmp_path_factory.mktemp("database")

    db_lookup = {}
    db_name = "test.db"
    db_url = f"sqlite:///{tmp_path}/{db_name}"

    engine = create_engine(db_url, echo=False)
    Entities.metadata.create_all(engine)
    Blobs.metadata.create_all(engine)

    connection = engine.connect()

    repository = RdbApi(connection)
    blob = BlobApi(connection)
    ######## add priors ########
    prior_key1 = repository.add_prior(
        "group", "key1", "function", ["paramA", "paramB"], [0.1, 0.2]
    )
    prior_key2 = repository.add_prior(
        "group", "key2", "function", ["paramA", "paramB"], [0.3, 0.4]
    )
    prior_key3 = repository.add_prior(
        "group", "key3", "function", ["paramA", "paramB"], [0.5, 0.6]
    )
    prior_A = repository.add_prior(
        "G", "A", "function", ["paramA", "paramB"], [0.5, 0.6]
    )
    repository.flush()

    ######## add ensemble ########
    ensemble = repository.add_ensemble(
        name="ensemble_name", priors=[prior_key1, prior_key2, prior_key3]
    )
    repository.flush()
    db_lookup["ensemble"] = ensemble.id
    db_lookup["ensemble_timestamp"] = ensemble.time_created
    ######## add parameteredefinitionss ########
    parameter_def_A_G = repository.add_parameter_definition(
        "A", "G", "ensemble_name", prior=prior_A
    )
    parameter_def_B_G = repository.add_parameter_definition("B", "G", "ensemble_name")
    parameter_def_key1_group = repository.add_parameter_definition(
        "key1", "group", "ensemble_name", prior=prior_key1
    )
    repository.flush()
    db_lookup["parameter_def_A_G"] = parameter_def_A_G.id
    db_lookup["parameter_def_key1_group"] = parameter_def_key1_group.id

    ######## add observations ########
    observation_one = repository.add_observation(
        name="observation_one",
        key_indexes_ref=add_blob([0, 3]),
        data_indexes_ref=add_blob([2, 3]),
        values_ref=add_blob([10.1, 10.2]),
        stds_ref=add_blob([1, 3]),
    )
    observation_one.add_attribute("region", "1")

    observation_two_first = repository.add_observation(
        name="observation_two_first",
        key_indexes_ref=add_blob(["2000-01-01 20:01:01"]),
        data_indexes_ref=add_blob([4]),
        values_ref=add_blob([10.3]),
        stds_ref=add_blob([2]),
    )

    observation_two_second = repository.add_observation(
        name="observation_two_second",
        key_indexes_ref=add_blob(["2000-01-02 20:01:01"]),
        data_indexes_ref=add_blob([5]),
        values_ref=add_blob([10.4]),
        stds_ref=add_blob([2.5]),
    )
    repository.flush()

    ######## add response definitions ########
    response_definition_one = repository.add_response_definition(
        name="response_one",
        indexes_ref=add_blob([3, 5, 8, 9]),
        ensemble_name=ensemble.name,
    )

    response_definition_two = repository.add_response_definition(
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
    repository.flush()
    db_lookup["response_defition_one"] = response_definition_one.id

    ######## observation response definition links ########
    obs_res_def_link = repository._add_observation_response_definition_link(
        observation_id=observation_one.id,
        response_definition_id=response_definition_one.id,
        active_ref=add_blob([True, False]),
        update_id=None,
    )

    repository._add_observation_response_definition_link(
        observation_id=observation_two_first.id,
        response_definition_id=response_definition_two.id,
        active_ref=add_blob([True]),
        update_id=None,
    )

    repository._add_observation_response_definition_link(
        observation_id=observation_two_second.id,
        response_definition_id=response_definition_two.id,
        active_ref=add_blob([True]),
        update_id=None,
    )
    repository.flush()

    ######## add realizations ########
    realization_0 = repository.add_realization(0, ensemble.name)
    realization_1 = repository.add_realization(1, ensemble.name)
    db_lookup["realization_0"] = realization_0.id

    def add_data(realization, response_def, ens):
        response_one = repository.add_response(
            name="response_one",
            values_ref=add_blob([11.1, 11.2, 9.9, 9.3]),
            realization_index=realization.index,
            ensemble_name=ens.name,
        )
        repository.flush()
        repository._add_misfit(200, obs_res_def_link.id, response_one.id)

        repository.add_response(
            name="response_two",
            values_ref=add_blob([12.1, 12.2, 11.1, 11.2, 9.9, 9.3]),
            realization_index=realization.index,
            ensemble_name=ens.name,
        )

        repository.add_parameter(
            "A", "G", add_blob(1), realization.index, "ensemble_name"
        )
        repository.add_parameter(
            "B", "G", add_blob(2), realization.index, "ensemble_name"
        )
        repository.add_parameter(
            "key1", "group", add_blob(2), realization.index, "ensemble_name"
        )

    add_data(realization_0, response_def=response_definition_one, ens=ensemble)
    add_data(realization_1, response_def=response_definition_one, ens=ensemble)

    repository.commit()
    ######## add blob #########
    data_blob = add_blob([0, 1, 2, 3])
    db_lookup["data_blob"] = data_blob

    yield (db_url, db_lookup)
