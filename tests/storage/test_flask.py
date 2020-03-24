import flask
from flask import Response
from flask import request
from tests.storage.app import FlaskWrapper
from tests.storage import populated_db, db_session, engine, tables
import pytest
#from ert_shared.storage.demo_api import data_definitions, get_data

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
    
    print(test_client.get("ensembles").data)
    #print(test_client.get("ensembles/1").data)
    