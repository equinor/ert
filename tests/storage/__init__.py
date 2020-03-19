import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ert_shared.storage import Entities, DataStore


@pytest.fixture(scope="session")
def engine():
    return create_engine("sqlite:///:memory:", echo=True)


@pytest.yield_fixture(scope="session")
def tables(engine):
    Entities.metadata.create_all(engine)
    DataStore.metadata.create_all(engine)
    yield
    Entities.metadata.drop_all(engine)
    DataStore.metadata.drop_all(engine)


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
