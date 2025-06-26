import json
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from ert.dark_storage.endpoints import router as endpoints_router
from ert.dark_storage.exceptions import ErtStorageError


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

    @staticmethod
    def render(content: Any) -> bytes:
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


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore
    yield


app = FastAPI(
    title="Dark Storage API",
    version="0.1.0",
    debug=True,
    default_response_class=JSONResponse,
    lifespan=lifespan,
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


@app.exception_handler(NotImplementedError)
async def not_implemented_handler(
    request: Request, exc: NotImplementedError
) -> JSONResponse:
    return JSONResponse({}, status_code=status.HTTP_501_NOT_IMPLEMENTED)


@app.get("/openapi.json", include_in_schema=False)
async def get_openapi() -> JSONResponse:
    return JSONResponse(app.openapi())


@app.get("/docs", include_in_schema=False)
async def get_swagger(req: Request) -> HTMLResponse:
    return get_swagger_ui_html(
        openapi_url="/openapi.json", title=f"{app.title} - Swagger UI"
    )


@app.get("/redoc", include_in_schema=False)
async def get_redoc(req: Request) -> HTMLResponse:
    return get_redoc_html(openapi_url="/openapi.json", title=f"{app.title} - Redoc")


@app.get("/")
async def root() -> RedirectResponse:
    return RedirectResponse("/docs")


@app.get("/healthcheck")
async def healthcheck() -> str:
    return "ALL OK!"


app.include_router(endpoints_router)
