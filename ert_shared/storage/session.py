from ert_shared.storage import Entities, Blobs
from sqlalchemy import create_engine
from sqlalchemy.orm.session import Session


class SessionFactory(object):
    def __init__(self):
        self._entities_session = None
        self._blobs_session = None

    def get_entities_session(self):
        if self._entities_session is None:
            engine = create_engine("sqlite:///entities.db", echo=False)
            Entities.metadata.create_all(engine)
            connection = engine.connect()
            self._entities_session = Session(bind=connection)
        return self._entities_session

    def get_blobs_session(self):
        if self._blobs_session is None:
            engine = create_engine("sqlite:///blobs.db", echo=False)
            Blobs.metadata.create_all(engine)
            connection = engine.connect()
            self._blobs_session = Session(bind=connection)
        return self._blobs_session


session_factory = SessionFactory()
