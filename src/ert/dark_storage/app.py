from ert_storage.app import JSONResponse
from ert_storage.app import app as ert_storage_app
from ert_storage.exceptions import ErtStorageError
from fastapi import FastAPI, Request, status
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.responses import HTMLResponse, RedirectResponse

from ert.dark_storage.endpoints import router as endpoints_router

app = FastAPI(
    title=ert_storage_app.title,
    version=ert_storage_app.version,
    debug=True,
    default_response_class=JSONResponse,
    # Disable documentation so we can replace it with ERT Storage's later
    openapi_url=None,
    docs_url=None,
    redoc_url=None,
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


@app.get("/openapi.json", include_in_schema=False)
async def get_openapi() -> JSONResponse:
    return JSONResponse(ert_storage_app.openapi())


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
