import logging

from fastapi import FastAPI
from starlette.testclient import TestClient

from ert.server.app import not_implemented_handler


def test_that_not_implemented_requests_are_logged_as_errors_with_traceback(caplog):
    app = FastAPI()
    app.add_exception_handler(NotImplementedError, not_implemented_handler)

    @app.get("/not_implemented")
    def not_implemented():
        raise NotImplementedError

    with caplog.at_level(logging.WARNING, logger="ert.server.app"):
        assert TestClient(app).get("/not_implemented").status_code == 501

    (record,) = caplog.records
    assert record.levelno == logging.ERROR
    assert "GET /not_implemented is not implemented" in record.message
    assert isinstance(record.exc_info[1], NotImplementedError)
