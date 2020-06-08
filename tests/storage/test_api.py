import sys

if sys.version_info.major >= 3:
    import schemathesis
    import pytest
    import hypothesis

    from ert_shared.storage.http_server import FlaskWrapper

    from tests.storage import db_info

    @pytest.fixture()
    def test_schema(db_info):
        populated_db, _ = db_info
        # Flask provides a way to test your application by exposing the Werkzeug test Client
        # and handling the context locals for you.
        flWrapper = FlaskWrapper(rdb_url=populated_db, blob_url=populated_db)
        # Establish an application context before running the tests.
        ctx = flWrapper.app.app_context()
        ctx.push()
        yield schemathesis.from_wsgi("/schema.json", flWrapper.app)
        ctx.pop()

    schema = schemathesis.from_pytest_fixture("test_schema")

    @schema.parametrize()
    @hypothesis.settings(
        suppress_health_check=[
            hypothesis.HealthCheck.filter_too_much,
            hypothesis.HealthCheck.too_slow,
        ]
    )
    def test_no_server_errors(case):
        response = case.call_wsgi()
        case.validate_response(response)
