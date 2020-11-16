import sys
from base64 import b64encode
from datetime import timedelta

import hypothesis
import pytest
import schemathesis
from ert_shared.storage import ERT_STORAGE
from ert_shared.storage.http_server import FlaskWrapper
from tests.storage import db_api, populated_database, initialize_databases


@pytest.fixture
def test_schema(db_api):
    # Flask provides a way to test your application by exposing the Werkzeug test Client
    # and handling the context locals for you.
    flWrapper = FlaskWrapper(secure=False, url=ERT_STORAGE.SQLALCHEMY_URL)
    # Establish an application context before running the tests.
    with flWrapper.app.app_context():
        yield schemathesis.from_wsgi("/schema.json", flWrapper.app)


schema = schemathesis.from_pytest_fixture("test_schema")


@schema.parametrize()
@hypothesis.settings(
    deadline=timedelta(milliseconds=1500),
    derandomize=True,
    suppress_health_check=[
        hypothesis.HealthCheck.filter_too_much,
        hypothesis.HealthCheck.too_slow,
    ],
)
def test_no_server_errors(case):
    response = case.call_wsgi()
    case.validate_response(response)
