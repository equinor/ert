import logging

from ert_shared.storage.model import Blobs, Entities
from sqlalchemy import create_engine
from sqlalchemy.orm.session import Session
from sqlalchemy.pool import NullPool


class SessionFactory(object):
    def __init__(self):
        self._entities_engine = None
        self._blobs_engine = None
        self._entities_db = "sqlite:///entities.db"
        self._blobs_db = "sqlite:///blobs.db"

    def get_entities_session(self):
        if self._entities_engine is None:
            msg = "Setting up session, using {}"
            logging.info(msg.format(self._entities_db))
            self._entities_engine = create_engine(self._entities_db, echo=False)
            Entities.metadata.create_all(self._entities_engine)
        connection = self._entities_engine.connect()
        return Session(bind=connection)

    def get_blobs_session(self):

        if self._blobs_engine is None:
            msg = "Setting up engine, using {}"
            logging.info(msg.format(self._blobs_db))
            self._blobs_engine = create_engine(self._blobs_db, echo=False)
            Blobs.metadata.create_all(self._blobs_engine)
        connection = self._blobs_engine.connect()
        return Session(bind=connection)


session_factory = SessionFactory()
