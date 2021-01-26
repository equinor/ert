import os
import sys
import textwrap
import asyncio
import atexit
import getpass
from typing import Generator
from fastapi import Depends
from pathlib import Path
from shutil import copy2, move

from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.engine import create_engine
from sqlalchemy.engine.url import make_url

import alembic
from alembic.config import Config
from alembic import script, migration


class ErtStorage:
    def __init__(self):
        self.Session = None
        self._cfg_file = Path(__file__).parent / "alembic.ini"
        self._db_file_name = "ert_storage.db"
        self._backup_dir = "backup"
        self.sqlalchemy_url = f"sqlite:///{self._db_file_name}"
        self._url = None
        self._engine = None
        self._cfg = None

    def initialize(
        self,
        project_path: Path = None,
        script_location=None,
        revision="head",
        testing=False,
    ):
        if project_path is None:
            project_path = Path.cwd()
        project_file = project_path / self._db_file_name

        if testing:
            database_file = project_file
            self._url = make_url(f"sqlite:///{project_path / self._db_file_name}")

            self._engine = create_engine(
                self._url, connect_args={"check_same_thread": False}
            )
            self.Session = sessionmaker(
                bind=self._engine, autocommit=False, autoflush=False
            )
        else:
            database_file = _tmp_db_path(project_path)
            if project_file.is_file():
                copy2(project_file, database_file)
            self._url = make_url(f"sqlite:///{database_file}")

            self._engine = create_engine(self._url)
            self.Session = sessionmaker(bind=self._engine)

        self.database_file = database_file
        self.project_file = project_file

        database_exists = os.path.isfile(database_file)

        self._cfg = Config(self._cfg_file)
        self._cfg.set_section_option("alembic", "sqlalchemy.url", str(self._url))

        if script_location is not None:
            self._cfg.set_section_option("alembic", "script_location", script_location)

        current_revision = self._db_revision()

        if self._revisions(index=0) == current_revision:
            print("Database up to date - continuing..")
            return

        if database_exists and current_revision is None:
            print(
                "Database found which is not under revision control. "
                "Renaming and initializing a new one."
            )
            if not os.path.isdir(self._backup_dir):
                os.mkdir(self._backup_dir)
            move(database_file, f"{self._backup_dir}/{database_file.name}")

        elif database_exists and self._revision_position(current_revision) > 0:
            print(
                "Database found, but is not updated. Will make backup and "
                "run migrations."
            )
            if not os.path.isdir(self._backup_dir):
                os.mkdir(self._backup_dir)
            copy2(
                database_file,
                f"{self._backup_dir}/{self._db_revision()}_{database_file.name}",
            )

        elif database_exists and self._revision_position(current_revision) == -1:
            sys.exit(self._error_msg(self._revisions(index=0)))

        with self._engine.begin() as connection:
            self._cfg.attributes["connection"] = connection
            alembic.command.upgrade(config=self._cfg, revision=revision)

    def shutdown(self):
        copy2(self.database_file, self.project_file)

    def _db_revision(self):
        with self._engine.begin() as conn:
            context = migration.MigrationContext.configure(conn)
            return context.get_current_revision()

    def _revisions(self, index=None):
        script_dir = script.ScriptDirectory.from_config(self._cfg)
        revisions = []
        for entry in script_dir.walk_revisions():
            revisions.append(entry.revision)

        if index is None:
            return revisions

        return revisions[index]

    def _revision_position(self, revision):
        revisions = self._revisions()
        if revision in revisions:
            return revisions.index(revision)
        return -1

    def _available_backup(self, revision=None):
        for file in os.listdir(self._backup_dir):
            if revision == file.split("_")[0]:
                return file

    def _error_msg(self, current_revision):
        backup_file = self._available_backup(current_revision)
        if backup_file is not None:
            return textwrap.fill(
                "Your database is configured for a newer "
                "version of ERT. However, there is a backup "
                "available supporting your version. To "
                "make us of that one instead, please "
                "replace your current '{}' with the backup "
                "'{}/{}'"
            ).format(self._db_file_name, self._backup_dir, backup_file)

        return textwrap.fill(
            "Your database is configured for a "
            "newer version of ERT. There is no "
            "backups supporting your current "
            "version. Please remove '{}' from your "
            "path to create a new storage, "
            "or switch to a newer version of ERT."
        ).format(self._db_file_name)


ERT_STORAGE = ErtStorage()


def _tmp_db_path(real_db_path: str) -> Path:
    if "XDG_RUNTIME_DIR" in os.environ:
        db_path = Path(os.environ["XDG_RUNTIME_DIR"]) / "ert"
        db_path.mkdir(exist_ok=True)
    else:
        db_path = Path(f"/tmp/ert-{getpass.getuser()}")
        db_path.mkdir(mode=0o700, exist_ok=True)

    db_path /= str(hash(real_db_path))
    db_path.mkdir(exist_ok=True)

    return db_path / f"ert_storage.db"


async def get_db() -> Generator[Session, None, None]:
    sess = ERT_STORAGE.Session()
    try:
        yield sess
        sess.commit()
    except BaseException:
        sess.rollback()
        raise
    finally:
        sess.close()


def Db() -> Depends:
    return Depends(get_db)
