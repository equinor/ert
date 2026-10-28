import logging

import pytest
from fastapi import FastAPI, status
from starlette.testclient import TestClient

from ert.server.app import ert_storage_error_handler, not_implemented_handler
from ert.server.exceptions import ErtStorageError, InternalServerError


class ConflictError(ErtStorageError):
    __status_code__ = status.HTTP_409_CONFLICT


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


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        pytest.param(InternalServerError("something broke"), 500, id="status_500"),
        pytest.param(ConflictError("already exists"), 409, id="status_below_500"),
    ],
)
def test_that_ert_storage_errors_are_logged_as_errors_with_traceback(
    caplog, error, expected_status
):
    app = FastAPI()
    app.add_exception_handler(ErtStorageError, ert_storage_error_handler)

    @app.get("/failing")
    def failing():
        raise error

    with caplog.at_level(logging.WARNING, logger="ert.server.app"):
        assert TestClient(app).get("/failing").status_code == expected_status

    (record,) = caplog.records
    assert record.levelno == logging.ERROR
    assert (
        f"GET /failing failed with status {expected_status}: {error.args[0]}"
        in record.message
    )
    assert record.exc_info[1] is error
