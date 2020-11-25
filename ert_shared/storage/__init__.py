import os
import textwrap
from pathlib import Path
from shutil import copy2, move

import alembic
from alembic import config, script, migration
from sqlalchemy.engine import create_engine
from sqlalchemy.engine.url import make_url
from sqlalchemy.orm import sessionmaker


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

    def initialize(self, url=None, script_location=None, revision="head"):
        if url is not None:
            self.sqlalchemy_url = url

        self._url = make_url(self.sqlalchemy_url)
        database_exists = os.path.isfile(self._url.database)

        self._engine = create_engine(self._url)
        self.Session = sessionmaker(bind=self._engine)

        self._cfg = config.Config(self._cfg_file)
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
            move(self._url.database, f"{self._backup_dir}/{self._url.database}")

        elif database_exists and self._revision_position(current_revision) > 0:
            print(
                "Database found, but is not updated. Will make backup and "
                "run migrations."
            )
            if not os.path.isdir(self._backup_dir):
                os.mkdir(self._backup_dir)
            copy2(
                self._url.database,
                f"{self._backup_dir}/{self._db_revision()}_{self._url.database}",
            )

        elif database_exists and self._revision_position(current_revision) == -1:
            raise Exception(self._error_msg(self._revisions(index=0)))

        with self._engine.begin() as connection:
            self._cfg.attributes["connection"] = connection
            alembic.command.upgrade(config=self._cfg, revision=revision)

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
