
import logging

from ert_shared.storage.model import Blobs, Entities
from sqlalchemy import create_engine
from sqlalchemy.orm.session import Session


class SessionFactory(object):
    def __init__(self):
        self._entities_session = None
        self._blobs_session = None
        self._entities_db = "sqlite:///entities.db"
        self._blobs_db = "sqlite:///blobs.db"

    def get_entities_session(self):
        if self._entities_session is None:
            msg = "Setting up session, using {}"
            logging.info(msg.format(self._entities_db))

            engine = create_engine(self._entities_db, echo=False)
            Entities.metadata.create_all(engine)
            connection = engine.connect()
            self._entities_session = Session(bind=connection)
        return self._entities_session

    def get_blobs_session(self):
        if self._blobs_session is None:
            msg = "Setting up session, using {}"
            logging.info(msg.format(self._blobs_db))

            engine = create_engine(self._blobs_db, echo=False)
            Blobs.metadata.create_all(engine)
            connection = engine.connect()
            self._blobs_session = Session(bind=connection)
        return self._blobs_session


session_factory = SessionFactory()
