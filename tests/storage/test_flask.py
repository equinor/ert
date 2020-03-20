import flask
from flask import Response
from flask import request
import pytest
#from ert_shared.storage.demo_api import data_definitions, get_data

@pytest.fixture(scope='module')
def test_client():
    flask_app = flask.Flask(__name__)
    @flask_app.route('/', methods=['GET'])
    def home():
        def hello():
            yield "Hello "
            yield "world"
        return Response(hello(), "text/plain")

    # Flask provides a way to test your application by exposing the Werkzeug test Client
    # and handling the context locals for you.
    testing_client = flask_app.test_client()
    # Establish an application context before running the tests.
    ctx = flask_app.app_context()
    ctx.push()
    yield testing_client
    ctx.pop()

def test_api(test_client):
    assert test_client.get().data == b"Hello world"