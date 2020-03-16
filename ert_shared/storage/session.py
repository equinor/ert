from ert_shared.storage import Base
from sqlalchemy import create_engine
from sqlalchemy.orm.session import Session


class SessionFactory(object):
    def __init__(self):
        self._session = None

    def get_session(self):
        if self._session is None:
            engine = create_engine("sqlite:///ert_storage.db", echo=False)
            Base.metadata.create_all(engine)
            connection = engine.connect()
            self._session = Session(bind=connection)
        return self._session


session_factory = SessionFactory()
