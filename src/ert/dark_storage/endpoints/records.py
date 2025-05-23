import io
import operator
from urllib.parse import unquote
from uuid import UUID, uuid4

import numpy as np
from fastapi import APIRouter, Body, Depends, File, Header, HTTPException
from fastapi.responses import Response

from ert.dark_storage import json_schema as js
from ert.dark_storage.common import (
    get_observation_keys_for_response,
    get_observations_for_obs_keys,
    get_storage,
)
from ert.storage import Storage

router = APIRouter(tags=["record"])

DEFAULT_STORAGE = Depends(get_storage)
DEFAULT_BODY = Body(...)
DEFAULT_FILE = File(...)
DEFAULT_HEADER = Header("application/json")


@router.get("/ensembles/{ensemble_id}/records/{response_name}/observations")
async def get_record_observations(
    *,
    storage: Storage = DEFAULT_STORAGE,
    ensemble_id: UUID,
    response_name: str,
) -> list[js.ObservationOut]:
    response_name = unquote(response_name)
    ensemble = storage.get_ensemble(ensemble_id)
    obs_keys = get_observation_keys_for_response(ensemble, response_name)
    obss = get_observations_for_obs_keys(ensemble, obs_keys)

    obss.sort(key=operator.itemgetter("name"))
    if not obss:
        return []

    return [
        js.ObservationOut(
            id=uuid4(),
            userdata={},
            errors=obs["errors"],
            values=obs["values"],
            x_axis=obs["x_axis"],
            name=obs["name"],
        )
        for obs in obss
    ]


@router.get("/ensembles/{ensemble_id}/records/{key}/std_dev")
def get_std_dev(
    *, storage: Storage = DEFAULT_STORAGE, ensemble_id: UUID, key: str, z: int
) -> Response:
    key = unquote(key)
    ensemble = storage.get_ensemble(ensemble_id)
    try:
        da = ensemble.calculate_std_dev_for_parameter_group(key)
    except ValueError as e:
        raise HTTPException(status_code=404, detail="Data not found") from e

    if z >= int(da.shape[2]):
        raise HTTPException(status_code=400, detail="Invalid z index")

    data_2d = da[:, :, z]

    buffer = io.BytesIO()
    np.save(buffer, data_2d)

    return Response(content=buffer.getvalue(), media_type="application/octet-stream")
