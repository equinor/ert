from tests.storage import populated_db, db_session, engine, tables
from ert_shared.storage.storage_api import StorageApi
import pytest
#from ert_shared.storage.demo_api import data_definitions, get_data
import json
import pprint


def test_response(populated_db):
    api = StorageApi(session=populated_db, blob_session=populated_db)
    api.response(1, "response_one", None)