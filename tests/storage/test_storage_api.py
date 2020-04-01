from tests.storage import populated_db, db_connection, engine, tables
from ert_shared.storage.rdb_api import RdbApi
from ert_shared.storage.blob_api import BlobApi
from ert_shared.storage.storage_api import StorageApi
import pytest
import json
import pprint


def test_response(populated_db):
    with StorageApi(rdb_connection=populated_db, blob_connection=populated_db) as api:
        schema = api.response(1, "response_one", None)
        assert len(schema["realizations"]) == 2
        assert len(schema["observation"]["data"]) == 4
        print("############### RESPONSE ###############")
        pprint.pprint(schema)


def test_ensembles(populated_db):
    with StorageApi(rdb_connection=populated_db, blob_connection=populated_db) as api:
        schema = api.ensembles()
        print("############### ENSEBMLES ###############")
        pprint.pprint(schema)


def test_ensemble(populated_db):
    with StorageApi(rdb_connection=populated_db, blob_connection=populated_db) as api:
        schema = api.ensemble_schema(1)
        print("############### ENSEMBLE ###############")
        pprint.pprint(schema)


def test_realization(populated_db):
    with StorageApi(rdb_connection=populated_db, blob_connection=populated_db) as api:
        schema = api.realization(ensemble_id=1, realization_idx=0, filter=None)
        print("############### REALIZATION ###############")
        pprint.pprint(schema)
