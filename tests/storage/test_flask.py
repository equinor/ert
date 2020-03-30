import flask
from flask import Response
from flask import request
from tests.storage.app import FlaskWrapper
from tests.storage import populated_db, db_session, engine, tables
import pytest
#from ert_shared.storage.demo_api import data_definitions, get_data
import json
import pprint

@pytest.fixture()
def test_client(populated_db):
    # Flask provides a way to test your application by exposing the Werkzeug test Client
    # and handling the context locals for you.
    flWrapper = FlaskWrapper(populated_db)
    testing_client = flWrapper.app.test_client()
    # Establish an application context before running the tests.
    ctx = flWrapper.app.app_context()
    ctx.push()
    yield testing_client
    ctx.pop()

def test_api(test_client):
    response = test_client.get("/ensembles")
    print(response.data)
    print(response.mimetype)
    ensembles = json.loads(response.data)
    
    
    for ens in ensembles["ensembles"]:
        print("########## ENSEMBLE #############")
        url = ens["ref_pointer"]
        ensemble = json.loads(test_client.get(url).data)
        pprint.pprint(ensemble)
        
        for real in ensemble['realizations']:
            print("########## ENSEMBLE - realization #############")
            realization = json.loads(test_client.get(real['ref_pointer']).data)
            pprint.pprint(realization)
        
            for response in realization['responses']:
                print("########## ENSEMBLE - realization - response #############")
                response_data = test_client.get(response['data_ref'])
                print(response_data.data)
        
        
        for response in ensemble['responses']:
            print("########## ENSEMBLE - response #############")
            response_data = test_client.get(response['ref_pointer'])
            print(response_data.data)


# def test_observation(test_client):
#     resp = test_client.get("/ensembles/ensemble_name")
#     ens = json.loads(resp.data)
#     expected = {
#         ("data_indexes", "2,3"),
#         ("key_indexes", "0,3"),
#         ("stds", "1,3"),
#         ("values", "10.1,10.2"),
#     }

#     actual = set()

#     for obs in ens["observations"]:
#         for data_ref, url in obs["data_refs"].items():
#             resp = test_client.get(url)
#             actual.add((data_ref, resp.data.decode("utf-8")))

#     assert actual == expected
