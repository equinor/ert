import json
from enum import Enum
from typing import Any

from fastapi import FastAPI, Request, Response, status
from fastapi.responses import RedirectResponse

from ert.dark_storage.endpoints import router as endpoints_router
from ert.dark_storage.exceptions import ErtStorageError
from ert.shared import __version__


class JSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder with support for Python 3.4 enums
    """

    def default(self, o: Any) -> Any:
        if isinstance(o, Enum):
            return o.name
        return super().default(o)


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
    title="ERT Storage API (dark storage)",
    version=__version__,
    debug=True,
    default_response_class=JSONResponse,
)


@app.on_event("startup")
async def initialize_libres_facade() -> None:
    # pylint: disable=import-outside-toplevel
    from ert.dark_storage.enkf import init_facade

    init_facade()


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


@app.exception_handler(NotImplementedError)
async def not_implemented_handler(
    request: Request, exc: NotImplementedError
) -> JSONResponse:
    return JSONResponse({}, status_code=status.HTTP_501_NOT_IMPLEMENTED)


@app.get("/")
async def root() -> RedirectResponse:
    return RedirectResponse("/docs")


@app.get("/healthcheck")
async def healthcheck() -> str:
    return "ALL OK!"


app.include_router(endpoints_router)
