import os
from contextlib import contextmanager
from pathlib import Path

from ert_shared.storage.blobs_model import Blobs
from ert_shared.storage.entities_model import Entities
from sqlalchemy.engine import create_engine
from sqlalchemy.orm import sessionmaker

import alembic
from alembic import config


class ErtStorage:
    def __init__(self):
        self.rdb_url = None
        self.blob_url = None

    def initialize(self, rdb_url=None, blob_url=None):
        if rdb_url == None:
            rdb_url = "sqlite:///{}/entities.db".format(os.getcwd())
        if blob_url == None:
            blob_url = "sqlite:///{}/blobs.db".format(os.getcwd())

        self.rdb_url = rdb_url
        self.blob_url = blob_url
        rdb_engine = create_engine(rdb_url)
        blob_engine = create_engine(blob_url)
        self.RdbSession = sessionmaker(bind=rdb_engine)
        self.BlobSession = sessionmaker(bind=blob_engine)

        self._upgrade_database(
            connection=rdb_engine.connect(), ini_section="alembic_rdb", url=self.rdb_url
        )
        self._upgrade_database(
            connection=blob_engine.connect(),
            ini_section="alembic_blob",
            url=self.blob_url,
        )

    def _upgrade_database(self, connection, ini_section, url, revision="head"):
        dirname = os.path.dirname(os.path.abspath(__file__))

        script_location = os.path.join(dirname, "alembic", ini_section)
        cfg_file = os.path.join(dirname, "alembic.ini")

        cfg = config.Config(cfg_file, ini_section=ini_section)
        cfg.set_section_option(ini_section, "sqlalchemy.url", url)
        cfg.set_section_option(ini_section, "script_location", script_location)
        cfg.attributes["connection"] = connection
        alembic.command.upgrade(config=cfg, revision=revision)


ERT_STORAGE = ErtStorage()
