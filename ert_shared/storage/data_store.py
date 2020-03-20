from ert_shared.storage import (
    Observation,
    Realization,
    Ensemble,
    ResponseDefinition,
    Response,
    ParameterDefinition,
    Parameter,
    DataFrame,
)
from sqlalchemy import create_engine
from sqlalchemy.orm import Bundle
from ert_shared.storage.session import session_factory


class DataStore:
    def __init__(self, session=None):

        if session is None:
            self._session = session_factory.get_session()
        else:
            self._session = session

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def commit(self):
        self._session.commit()
    
    def flush(self):
        self._session.flush()

    def rollback(self):
        self._session.rollback()

    def close(self):
        self._session.close()

    def add_data_frame(self, data):
        data_frame = DataFrame(data=data)
        self._session.add(data_frame)
        return data_frame

    def get_data_frame(self, id):
        return self._session.query(DataFrame).get(id)
