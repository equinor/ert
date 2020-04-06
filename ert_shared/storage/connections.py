import logging

from ert_shared.storage.model import Blobs, Entities
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool


def get_rdb_connection(url, pragma_foreign_keys=True):
    msg = "Setting up session, using {}"
    logging.info(msg.format(url))
    engine = create_engine(url, echo=False)
    if pragma_foreign_keys:
        engine.execute("pragma foreign_keys=on")
    Entities.metadata.create_all(engine)
    return engine.connect()


def get_blob_connection(url, pragma_foreign_keys=True):
    msg = "Setting up engine, using {}"
    logging.info(msg.format(url))
    engine = create_engine(url, echo=False)
    if pragma_foreign_keys:
        engine.execute("pragma foreign_keys=on")
    Blobs.metadata.create_all(engine)
    return engine.connect()
