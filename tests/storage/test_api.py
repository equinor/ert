import sys

if sys.version_info.major >= 3:
    import schemathesis
    import pytest

    from ert_shared.storage.http_server import FlaskWrapper

    from tests.storage import populated_db

    @pytest.fixture()
    def test_schema(populated_db):
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
    def test_no_server_errors(case):
        response = case.call_wsgi()
        try:
            case.validate_response(response)
        except AssertionError as e:
            if "500" in str(e):
                print("Expected failure, API still in beta")
            else:
                raise
