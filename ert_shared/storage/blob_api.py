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
    def __init__(self, session):
        self._session = session

    def add_blob(self, data):
        data_frame = ErtBlob(data=data)
        self._session.add(data_frame)
        self._session.flush()
        return data_frame

    def get_blob(self, id):
        return self._session.query(ErtBlob).get(id)

    def get_blobs(self, ids):
        if not isinstance(ids, list):
            ids = [ids]

        return (
            self._session.query(ErtBlob)
            .filter(ErtBlob.id.in_(ids))
            .yield_per(1)
            .enable_eagerloads(False)
        )
