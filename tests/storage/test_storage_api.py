from tests.storage import populated_db, db_session, engine, tables
from ert_shared.storage.rdb_api import RdbApi
from ert_shared.storage.blob_api import BlobApi
from ert_shared.storage.storage_api import StorageApi
import pytest
import json
import pprint

def resolve_data_uri(BASE_URL, struct):
    if isinstance(struct, list):
        for item in struct:
            resolve_data_uri(BASE_URL, item)
    elif isinstance(struct, dict):
        for key, val in struct.copy().items():
            if key == "data_ref":
                url = "{}/data/{}".format(BASE_URL, val)
                struct["data_url"] = url
            else:
                resolve_data_uri(BASE_URL, val)

def resolve_ref_uri(BASE_URL, struct):
    if isinstance(struct, list):
        for item in struct:
            resolve_ref_uri(BASE_URL, item)
    elif isinstance(struct, dict):
        for key, val in struct.copy().items():
            if "_ref" in key and "data" not in key:
                url = "{}/{}".format(BASE_URL, val)
                struct["ref_url"] = url
            else:
                resolve_ref_uri("{}/{}".format(BASE_URL, key), val)

def test_response(populated_db):
    with RdbApi(populated_db) as rdb_api, BlobApi(populated_db) as blob_api:
        api = StorageApi(rdb_api=rdb_api, blob_api=blob_api)        
        schema = api.response(1, "response_one", None)
        assert len(schema['realizations']) == 2
        assert len(schema["observation"]["data"]) == 4
        print("############### RESPONSE ###############")
        pprint.pprint(schema)
        print("--------------- after --------------")
        resolve_data_uri("http://localhost", schema)
        resolve_ref_uri("http://localhost/ensembles/1/response/response_one", schema)
        pprint.pprint(schema)

def test_ensembles(populated_db):
    with RdbApi(populated_db) as rdb_api, BlobApi(populated_db) as blob_api:
        api = StorageApi(rdb_api=rdb_api, blob_api=blob_api)        
        schema = api.ensembles()
        print("############### ENSEBMLES ###############")
        pprint.pprint(schema)
        print("--------------- after --------------")
        resolve_data_uri("http://localhost", schema)
        resolve_ref_uri("http://localhost/ensembles", schema)
        pprint.pprint(schema)

def test_ensemble(populated_db):
    with RdbApi(populated_db) as rdb_api, BlobApi(populated_db) as blob_api:
        api = StorageApi(rdb_api=rdb_api, blob_api=blob_api)        
        schema = api.ensemble_schema(1)
        print("############### ENSEMBLE ###############")
        pprint.pprint(schema)
        print("--------------- after --------------")
        resolve_data_uri("http://localhost", schema)
        resolve_ref_uri("http://localhost/ensembles/1", schema)
        pprint.pprint(schema)

def test_realization(populated_db):
    with RdbApi(populated_db) as rdb_api, BlobApi(populated_db) as blob_api:
        api = StorageApi(rdb_api=rdb_api, blob_api=blob_api)        
        schema = api.realization(ensemble_id=1, realization_idx=0, filter=None)
        print("############### REALIZATION ###############")
        pprint.pprint(schema)
        print("--------------- after --------------")
        resolve_data_uri("http://localhost", schema)
        resolve_ref_uri("http://localhost/ensembles/1", schema)
        pprint.pprint(schema)
