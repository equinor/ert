import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from ert_shared.storage.repository import ErtRepository

from ert_shared.storage import Entities, Blobs


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
            values=[10.1, 10.2],
            stds=[1, 3],
        )

        repository.add_response_definition(
            name="response_one",
            indexes=[0, 1],
            ensemble_name=ensemble.name,
            observation_name=observation.name
        )

        repository.add_response(
            name="response_one",
            values=[11.1, 11.2],
            realization_index=realization.index,
            ensemble_name=ensemble.name,
        )

        repository.add_response_definition(
            name="response_two",
            indexes=[0, 1],
            ensemble_name=ensemble.name,
        )

        repository.add_response(
            name="response_two",
            values=[12.1, 12.2],
            realization_index=realization.index,
            ensemble_name=ensemble.name,
        )
        repository.add_parameter_definition("A", "G", "ensemble_name")
        repository.add_parameter_definition("B", "G", "ensemble_name")
        repository.add_parameter("A", "G", 1, 0, "ensemble_name")
        repository.add_parameter("B", "G", 2, 0, "ensemble_name")
        repository.commit()

    yield db_session
