from pathlib import Path
from sqlalchemy.engine import create_engine
from sqlalchemy.orm import sessionmaker

import alembic
from alembic import config


class ErtStorage:
    SQLALCHEMY_URL = "sqlite:///ert_storage.db"

    def initialize(self, url=None):
        if url is not None:
            self.SQLALCHEMY_URL = url

        engine = create_engine(self.SQLALCHEMY_URL)
        self.Session = sessionmaker(bind=engine)

        cfg = config.Config(Path(__file__).parent / "alembic.ini")
        cfg.set_section_option("alembic", "sqlalchemy.url", self.SQLALCHEMY_URL)
        cfg.attributes["connection"] = engine.connect()
        alembic.command.upgrade(config=cfg, revision="head")


ERT_STORAGE = ErtStorage()
