from ert_shared.storage.model import (
    Observation,
    Realization,
    Ensemble,
    ResponseDefinition,
    Response,
    ParameterDefinition,
    Parameter,
    ErtBlob,
)
from sqlalchemy import create_engine
from sqlalchemy.orm import Bundle
from sqlalchemy.orm.session import Session


class BlobApi:
    def __init__(self, connection):
        self._session = Session(bind=connection)

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

    def close_connection(self):
        self._session.connection().close()

    def add_blob(self, data):
        data_frame = ErtBlob(data=data)
        self._session.add(data_frame)
        return data_frame

    def get_blob(self, id):
        return self._session.query(ErtBlob).get(id)
