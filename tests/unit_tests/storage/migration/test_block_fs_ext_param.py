import logging

import pytest

import storage.migration.block_fs as bf
from ert.config import ErtConfig
from storage.local_storage import local_storage_set_ert_config


@pytest.fixture(scope="module")
def enspath(block_storage_path):
    return block_storage_path / "ext_param/storage"


@pytest.fixture(scope="module")
def ert_config(block_storage_path):
    return ErtConfig.from_file(str(block_storage_path / "ext_param/config.ert"))


@pytest.fixture(scope="module", autouse=True)
def set_ert_config(ert_config):
    yield local_storage_set_ert_config(ert_config)
    local_storage_set_ert_config(None)


def test_migration_failure(storage, enspath, caplog, monkeypatch):
    """Run migration but fail due to missing config data. Expected behaviour is
    for the error to be logged but no exception be propagated.

    """
    monkeypatch.setattr(storage, "open_storage", lambda: storage)

    # Sanity check: no ensembles are created before migration
    assert list(storage.ensembles) == []

    with caplog.at_level(logging.WARNING, logger="storage.migration.block_fs"):
        bf._migrate_case_ignoring_exceptions(storage, enspath / "sim_fs")

    # No ensembles were created due to failure
    assert list(storage.ensembles) == []

    # Warnings are in caplog
    assert len(caplog.records) == 1
    assert caplog.records[0].message == (
        "Exception occurred during migration of BlockFs case 'sim_fs': "
        "Migrating EXT_PARAM is not supported"
    )
