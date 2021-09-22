import json
from enum import Enum
from typing import Any
from fastapi import FastAPI, Request, status
from fastapi.responses import Response, RedirectResponse
from starlette.graphql import GraphQLApp

from ert_storage.endpoints import router as endpoints_router
from ert_storage.graphql import router as graphql_router
from ert_storage.exceptions import ErtStorageError

from sqlalchemy.orm.exc import NoResultFound


class JSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder with support for Python 3.4 enums
    """

    def default(self, obj: Any) -> Any:
        if isinstance(obj, Enum):
            return obj.name
        return super().default(obj)


class JSONResponse(Response):
    """A replacement for Starlette's JSONResponse that permits NaNs."""

    media_type = "application/json"

    def render(self, content: Any) -> bytes:
        return (
            JSONEncoder(
                ensure_ascii=False,
                allow_nan=True,
                indent=None,
                separators=(",", ":"),
            )
            .encode(content)
            .encode("utf-8")
        )


app = FastAPI(
    title="ERT Storage API",
    version="0.1.2",
    debug=True,
    default_response_class=JSONResponse,
)


@app.on_event("startup")
async def initialize_database() -> None:
    from ert_storage.database import engine, IS_SQLITE, HAS_AZURE_BLOB_STORAGE
    from ert_storage.database_schema import Base

    if IS_SQLITE:
        # Our SQLite backend doesn't support migrations, so create the database on the fly.
        Base.metadata.create_all(bind=engine)
    if HAS_AZURE_BLOB_STORAGE:
        from ert_storage.database import create_container_if_not_exist

        await create_container_if_not_exist()


@app.exception_handler(NoResultFound)
async def sqlalchemy_exception_handler(
    request: Request, exc: NoResultFound
) -> JSONResponse:
    """Automatically catch and convert an SQLAlchemy NoResultFound exception (when
    using `.one()`, for example) to an HTTP 404 message
    """
    return JSONResponse(
        {"detail": "Item not found"}, status_code=status.HTTP_404_NOT_FOUND
    )


@app.exception_handler(ErtStorageError)
async def ert_storage_error_handler(
    request: Request, exc: ErtStorageError
) -> JSONResponse:
    return JSONResponse(
        {
            "detail": {
                **request.query_params,
                **request.path_params,
                **exc.args[1],
                "error": exc.args[0],
            }
        },
        status_code=exc.__status_code__,
    )


@app.get("/")
async def root() -> RedirectResponse:
    return RedirectResponse("/docs")


@app.get("/healthcheck")
async def healthcheck() -> str:
    return "ALL OK!"


app.include_router(endpoints_router)
app.include_router(graphql_router)
