
import logging

from ert_shared.storage.model import Blobs, Entities
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool


def get_rdb_connection(url="sqlite:///entities.db"):
    msg = "Setting up session, using {}"
    logging.info(msg.format(url))
    entities_engine = create_engine(url, echo=False)
    Entities.metadata.create_all(entities_engine)
    return entities_engine.connect()

def get_blob_connection(url="sqlite:///blobs.db"):
    msg = "Setting up engine, using {}"
    logging.info(msg.format(url))
    blobs_engine = create_engine(url, echo=False)
    Blobs.metadata.create_all(blobs_engine)
    return blobs_engine.connect()
