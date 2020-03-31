from tests.storage import populated_db, db_session, engine, tables
from ert_shared.storage.storage_api import StorageApi
import pytest
import json
import pprint


def test_response(populated_db):
    api = StorageApi(rdb_session=populated_db, blob_session=populated_db)
    schema = api.response(1, "response_one", None)
    assert len(schema['realizations']) == 2
    assert len(schema["observation"]["data"]) == 3
    pprint.pprint(schema)
