import math
from pydantic import BaseModel

from ert_shared.storage.app import app


class _NanModel(BaseModel):
    value: float


@app.get("/test_nan_json")
async def _nan_json():
    return math.nan


@app.get("/test_nan_pydantic", response_model=_NanModel)
async def _nan_pydantic():
    return _NanModel(value=math.nan)


def test_nan(app_client):
    """
    The JSONResponse as found in Starlette explicitly disallows NaNs when
    encoding. Both the direct return of JSON-able objects and Pydantic use
    JSONResponse.
    """
    resp = app_client.get("/test_nan_json")
    assert math.isnan(resp.json())

    resp = app_client.get("/test_nan_pydantic")
    assert math.isnan(resp.json()["value"])
