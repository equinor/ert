import os
import sys
from fastapi import FastAPI, Request, status
from fastapi.exceptions import HTTPException
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBasic

from ert_shared.storage import json_schema as js
from ert_shared.storage.db import ERT_STORAGE
from ert_shared.storage.endpoints import router

from sqlalchemy.orm.exc import NoResultFound


app = FastAPI(name="ERT Storage API", debug=True)
security = HTTPBasic()


@app.on_event("startup")
async def prepare_database():
    ERT_STORAGE.initialize()


@app.middleware("http")
async def check_authorization(request: Request, call_next):
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
    return {}


@app.get("/healthcheck", response_model=js.Healthcheck)
async def healthcheck():
    from datetime import datetime

    return {"date": datetime.now().isoformat()}


app.include_router(router)
