import os

import pytest
from ert_shared.storage.blob_api import BlobApi
from ert_shared.storage.model import Blobs, Entities
from ert_shared.storage.rdb_api import RdbApi
from sqlalchemy import create_engine
from sqlalchemy.orm import Session


@pytest.fixture(scope="session")
def engine():
    return create_engine("sqlite:///:memory:", echo=False)


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

    repository.add_response_definition(
        name="response_one",
        indexes_ref=add_blob([0, 1]),
        ensemble_name=ensemble.name,
        observation_name=observation.name,
    )

    repository.add_response_definition(
        name="response_two", indexes_ref=add_blob([0, 1]), ensemble_name=ensemble.name,
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
