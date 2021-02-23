import os
import sys
import json
from typing import Any
from fastapi import FastAPI, Request, status
from fastapi.exceptions import HTTPException
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import Response, RedirectResponse
from fastapi.security import HTTPBasic

from ert_shared.storage import json_schema as js
from ert_shared.storage.db import ERT_STORAGE
from ert_shared.storage.endpoints import router

from sqlalchemy.orm.exc import NoResultFound


try:
    from importlib.metadata import version as _version

    ert_version = _version("ert")
except ModuleNotFoundError:
    from pkg_resources import get_distribution as _get_distribution

    ert_version = _get_distribution("ert").version


class JSONResponse(Response):
    """A replacement for Starlette's JSONResponse that permits NaNs."""

    media_type = "application/json"

    def render(self, content: Any) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=True,
            indent=None,
            separators=(",", ":"),
        ).encode("utf-8")


app = FastAPI(
    title="ERT Storage API",
    description="Experimental new database-based storage system for ERT",
    version=ert_version,
    default_response_class=JSONResponse,
    debug=os.getenv("ERT_STORAGE_DEBUG"),
)
security = HTTPBasic()


@app.on_event("startup")
async def prepare_database():
    try:
        ERT_STORAGE.initialize()
    except SystemExit as exc:
        # SystemExit is caught by Startlette, which prints a stack trace. This
        # is not what we want. We want a clean exit. As such, we use os._exit,
        # which doesn't clean up very well after itself, but it should be fine
        # here.
        if isinstance(exc.code, str):
            print(exc.code, file=sys.stderr)
        elif isinstance(exc.code, int):
            os._exit(exc.code)
        os._exit(1)


@app.on_event("shutdown")
async def shutdown_database():
    ERT_STORAGE.shutdown()


@app.middleware("http")
async def check_authorization(request: Request, call_next):
    if os.getenv("ERT_STORAGE_DEBUG"):
        return await call_next(request)

    api_token = os.getenv("ERT_AUTHTOKEN")
    if not api_token:
        return await call_next(request)

    try:
        credentials = await security(request)
    except HTTPException as e:
        # HTTPBasic throws an HTTPException when it can't validate. However,
        # we're a middleware and not an endpoint and as such the exception
        # handlers don't trigger. Do it manually
        return await http_exception_handler(request, e)

    if not (credentials.username == "__token__" and credentials.password == api_token):
        return JSONResponse(
            {"detail": "Invalid credentials"},
            status_code=status.HTTP_403_FORBIDDEN,
            headers={"WWW-Authenticate": "Basic"},
        )

    return await call_next(request)


@app.exception_handler(NoResultFound)
async def sqlalchemy_exception_handler(request: Request, exc: NoResultFound):
    """Automatically catch and convert an SQLAlchemy NoResultFound exception (when
    using `.one()`, for example) to an HTTP 404 message
    """
    return JSONResponse(
        {"detail": "Item not found"}, status_code=status.HTTP_404_NOT_FOUND
    )


@app.get("/")
async def root():
    return RedirectResponse("/docs")


@app.get("/healthcheck", response_model=js.Healthcheck)
async def healthcheck():
    from datetime import datetime

    return {"date": datetime.now().isoformat()}


app.include_router(router)
