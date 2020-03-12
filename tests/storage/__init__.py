import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from ert_shared.storage.repository import ErtRepository

from ert_shared.storage import Entities, Blobs


@pytest.fixture(scope="session")
def engine():
    return create_engine("sqlite:///:memory:", echo=True)


@pytest.yield_fixture(scope="session")
def tables(engine):
    Entities.metadata.create_all(engine)
    Blobs.metadata.create_all(engine)
    yield
    Entities.metadata.drop_all(engine)
    Blobs.metadata.drop_all(engine)


@pytest.yield_fixture
def db_session(engine, tables):
    """Returns an sqlalchemy session, and after the test tears down everything properly."""
    connection = engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)

    yield session

    session.close()
    transaction.rollback()
    connection.close()

@pytest.yield_fixture
def populated_db(db_session):
    with ErtRepository(db_session) as repository:
        ensemble = repository.add_ensemble(name="ensemble_name")

        realization = repository.add_realization(0, ensemble.name)

        observation = repository.add_observation(
            name="observation_one",
            key_indexes=[0, 3],
            data_indexes=[0, 3],
            values=[22.1, 44.2],
            stds=[1, 3],
        )
        repository.commit()

        repository.add_response(
            name="response_one",
            values=[22.1, 44.2],
            indexes=[0, 1],
            realization_index=realization.index,
            ensemble_name=ensemble.name,
            observation_id=observation.id
        )
        repository.commit()

        yield repository