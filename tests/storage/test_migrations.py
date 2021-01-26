import os
from pathlib import Path

import pytest
from sqlalchemy.engine import create_engine

from ert_shared.storage.db import ErtStorage
from ert_shared.storage.database_schema import Entity


def test_initialize_from_scratch(tmpdir):
    script_location = str(Path(__file__).parent / "migrations_testing")
    with tmpdir.as_cwd():
        storage = ErtStorage()
        assert not os.path.isfile(storage._db_file_name)
        storage.initialize(script_location=script_location)
        storage.shutdown()
        assert os.path.isfile(storage._db_file_name)


def test_initialize_db_already_exists_without_versioning(tmpdir):
    script_location = str(Path(__file__).parent / "migrations_testing")
    with tmpdir.as_cwd():
        # Just create db with all tables not using alembic
        storage = ErtStorage()
        engine = create_engine(storage.sqlalchemy_url)
        Entity.metadata.create_all(engine)

        storage.initialize(script_location=script_location)
        assert os.path.isfile(f"{storage._backup_dir}/{storage._db_file_name}")
        assert os.path.isfile(storage._db_file_name)


def test_db_backup_before_upgrade(tmpdir):
    script_location = str(Path(__file__).parent / "migrations_testing")
    with tmpdir.as_cwd():
        storage = ErtStorage()
        storage.initialize(script_location=script_location, revision="11d6bbf0a926")
        storage.shutdown()
        initial_revision = storage._db_revision()

        storage.initialize(script_location=script_location)
        storage.shutdown()
        last_revision = storage._db_revision()

        assert initial_revision != last_revision
        assert os.path.isfile(storage._db_file_name)
        assert os.path.isfile(
            f"{storage._backup_dir}/{initial_revision}_{storage._db_file_name}"
        )


def test_ert_too_old_with_backup(tmpdir):
    with tmpdir.as_cwd():
        # Simulate ERT Storage ran with stable
        script_location = str(Path(__file__).parent / "migrations_stable")
        storage = ErtStorage()
        storage.initialize(script_location=script_location)
        storage.shutdown()
        stable_revision = storage._db_revision()

        # Switch to testing and upgrade database
        script_location = str(Path(__file__).parent / "migrations_testing")
        storage.initialize(script_location=script_location)
        storage.shutdown()
        testing_revision = storage._db_revision()

        assert stable_revision != testing_revision
        assert os.path.isfile(storage._db_file_name)
        assert os.path.isfile(
            f"{storage._backup_dir}/{stable_revision}_{storage._db_file_name}"
        )

        # Switch back to stable. Database is now too new
        script_location = str(Path(__file__).parent / "migrations_stable")
        with pytest.raises(SystemExit) as excinfo:
            storage.initialize(script_location=script_location)
        assert storage._error_msg(stable_revision) in str(excinfo.value)


def test_ert_too_old_without_backup(tmpdir):
    with tmpdir.as_cwd():
        # Simulate ERT Storage ran with testing
        script_location = str(Path(__file__).parent / "migrations_testing")
        storage = ErtStorage()
        storage.initialize(script_location=script_location)
        storage.shutdown()
        testing_revision = storage._db_revision()

        # Switch to bleeding and upgrade database
        script_location = str(Path(__file__).parent / "migrations_bleeding")
        storage.initialize(script_location=script_location)
        storage.shutdown()
        bleeding_revision = storage._db_revision()

        assert testing_revision != bleeding_revision
        assert os.path.isfile(storage._db_file_name)
        assert os.path.isfile(
            f"{storage._backup_dir}/{testing_revision}_{storage._db_file_name}"
        )

        # Switch back to stable. Database is now too new and no backup for stable
        script_location = str(Path(__file__).parent / "migrations_stable")
        with pytest.raises(SystemExit) as excinfo:
            storage.initialize(script_location=script_location)
        assert storage._error_msg("rev-for-test-purpose") in str(excinfo.value)
