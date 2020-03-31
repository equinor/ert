from tests.storage import populated_db, db_session, engine, tables
from ert_shared.storage.rdb_api import RdbApi
from ert_shared.storage.blob_api import BlobApi
from ert_shared.storage.storage_api import StorageApi
import pytest
import json
import pprint


def test_response(populated_db):
    with RdbApi(populated_db) as rdb_api, BlobApi(populated_db) as blob_api:
        api = StorageApi(rdb_api=rdb_api, blob_api=blob_api)
        schema = api.response(1, "response_one", None)
        assert len(schema["realizations"]) == 2
        assert len(schema["observation"]["data"]) == 3
        pprint.pprint(schema)
