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


@pytest.yield_fixture(scope="session")
def tables(engine):
    Entities.metadata.create_all(engine)
    Blobs.metadata.create_all(engine)
    yield
    Entities.metadata.drop_all(engine)
    Blobs.metadata.drop_all(engine)


@pytest.yield_fixture
def db_connection(engine, tables):
    """Returns an sqlalchemy session, and after the test tears down everything properly."""
    connection = engine.connect()
    transaction = connection.begin()

    yield connection

    transaction.rollback()
    connection.close()


@pytest.yield_fixture
def populated_db(tmpdir):
    db_name = "test.db"
    db_url = "sqlite:///{}/{}".format(tmpdir, db_name)

    engine = create_engine(db_url, echo=False)
    Entities.metadata.create_all(engine)
    Blobs.metadata.create_all(engine)

    connection = engine.connect()

    repository = RdbApi(connection)
    blob = BlobApi(connection)

    ensemble = repository.add_ensemble(name="ensemble_name")

    realization = repository.add_realization(0, ensemble.name)
    realization2 = repository.add_realization(1, ensemble.name)

    def add_blob(data):
        ret = blob.add_blob(data)
        blob.flush()
        return ret.id

    observation = repository.add_observation(
        name="observation_one",
        key_indexes_ref=add_blob([0, 3]),
        data_indexes_ref=add_blob([2, 3]),
        values_ref=add_blob([10.1, 10.2]),
        stds_ref=add_blob([1, 3]),
    )

    response_definition = repository.add_response_definition(
        name="response_one", indexes_ref=add_blob([3, 5]), ensemble_name=ensemble.name,
    )
    repository.flush()

    repository._add_observation_response_definition_link(
        observation_id=observation.id, response_definition_id=response_definition.id
    )

    observation_one = repository.add_observation(
        name="observation_two_first",
        key_indexes_ref=add_blob(["2000-01-01 20:01:01"]),
        data_indexes_ref=add_blob([4]),
        values_ref=add_blob([10.3]),
        stds_ref=add_blob([2]),
    )

    observation_two = repository.add_observation(
        name="observation_two_second",
        key_indexes_ref=add_blob(["2000-01-02 20:01:01"]),
        data_indexes_ref=add_blob([5]),
        values_ref=add_blob([10.4]),
        stds_ref=add_blob([2.5]),
    )

    response_two_definition = repository.add_response_definition(
        name="response_two",
        indexes_ref=add_blob(["2000-01-01 20:01:01", "2000-01-02 20:01:01"]),
        ensemble_name=ensemble.name,
    )

    repository.flush()

    repository._add_observation_response_definition_link(
        observation_id=observation_one.id,
        response_definition_id=response_two_definition.id,
    )

    repository._add_observation_response_definition_link(
        observation_id=observation_two.id,
        response_definition_id=response_two_definition.id,
    )

    repository.add_parameter_definition("A", "G", "ensemble_name")
    repository.add_parameter_definition("B", "G", "ensemble_name")

    def add_data(realization):
        repository.add_response(
            name="response_one",
            values_ref=add_blob([11.1, 11.2]),
            realization_index=realization.index,
            ensemble_name=ensemble.name,
        )

        repository.add_response(
            name="response_two",
            values_ref=add_blob([12.1, 12.2]),
            realization_index=realization.index,
            ensemble_name=ensemble.name,
        )

        repository.add_parameter(
            "A", "G", add_blob(1), realization.index, "ensemble_name"
        )
        repository.add_parameter(
            "B", "G", add_blob(2), realization.index, "ensemble_name"
        )

    add_data(realization)
    add_data(realization2)

    repository.commit()
    yield db_url
